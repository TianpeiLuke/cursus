---
tags:
  - project
  - implementation
  - multi_strategy_path_resolution
  - deployment_portability
  - config_portability
  - step_builders
  - universal_deployment
keywords:
  - multi strategy path resolution
  - deployment scenario detection
  - universal path portability
  - configuration simplification
  - runtime context adaptation
  - bundled deployment
  - monorepo deployment
  - pip installed deployment
topics:
  - multi-strategy path resolution implementation
  - deployment scenario detection system
  - universal configuration portability
  - step builder modernization
  - deployment agnostic architecture
language: python
date of note: 2025-09-22
---

# Multi-Strategy Deployment Path Resolution Implementation Plan

## Executive Summary

This implementation plan provides a detailed roadmap for implementing the **Multi-Strategy Deployment Path Resolution System** that automatically detects deployment scenarios and applies the appropriate path resolution strategy. This implementation **reverts and replaces** the previous portable path approach from the 2025-09-20 plan, which was proven unable to support all three deployment scenarios.

### Key Objectives

#### **Primary Objectives**
- **Implement Multi-Strategy Path Resolution**: Automatic detection of bundled, monorepo, and pip-installed scenarios
- **Enable Universal Deployment**: Same config files work across Lambda/MODS, development, and pip-installed environments
- **Remove Failed Portable Path Infrastructure**: Clean removal of `portable_source_dir` and `portable_processing_source_dir` fields and related methods
- **Maintain Zero Breaking Changes**: All existing code continues working without modification

#### **Secondary Objectives**
- **Enhance Configuration Fields**: Add `project_root_folder` field for bundled deployments
- **Implement Robust Fallbacks**: Automatic fallback mechanisms for each scenario
- **Enable Future Extensibility**: Foundation for additional deployment scenarios
- **Improve Developer Experience**: Transparent operation with enhanced error handling

### Strategic Impact

- **Universal Configuration Sharing**: Developers can share configs across all environments
- **Lambda/MODS Deployment Success**: Fixes critical MODS pipeline deployment failures
- **Development Environment Compatibility**: Works seamlessly in monorepo development
- **PyPI Package Compatibility**: Supports pip-installed cursus with user projects
- **Future-Ready Architecture**: Foundation for additional deployment scenarios

## Problem Analysis

### **Critical Issues with Previous Approach** (from 2025-09-20 Plan)

#### **❌ Failed Portable Path Infrastructure**
The previous implementation attempted to create "portable" paths by converting absolute paths to relative paths, but this approach failed because:

1. **Context Mismatch**: Paths calculated in development context failed in deployment context
2. **Single Strategy Limitation**: One-size-fits-all approach couldn't handle diverse deployment architectures
3. **Lambda Sibling Directory Issue**: Failed to handle Lambda's sibling directory structure
4. **Working Directory Dependency**: Relied on working directory which varies across deployments

#### **❌ Problematic Fields and Methods to Remove**
- **`portable_source_dir`** property - generates incorrect paths
- **`portable_processing_source_dir`** property - fails in Lambda deployments
- **`_portable_source_dir`** private field - stores incorrect relative paths
- **`_portable_processing_source_dir`** private field - context-dependent failures
- **`_convert_to_relative_path()`** method - fundamentally flawed approach
- **`get_portable_script_path()`** method - produces non-working paths

### **Solution Architecture** (from Multi-Strategy Design)

#### **Deployment Scenario Detection**
- **Bundled (Lambda/MODS)**: Detect sibling directories to cursus package
- **Monorepo (Development)**: Detect `src/cursus` structure pattern
- **Pip-Installed (Separated)**: Default when neither above pattern found

#### **Strategy-Specific Path Resolution**
- **Bundled**: `package_root + project_root_folder + source_dir`
- **Monorepo**: `project_root + source_dir`
- **Pip-Installed**: `discovered_user_project + source_dir`

## Implementation Phases

### **Phase 1: Remove Failed Portable Path Infrastructure** (Week 1)

#### **Objective**: Clean removal of all portable path related code that was proven to fail

#### **Day 1-2: Remove Portable Path Fields and Properties**

**Target File**: `src/cursus/core/base/config_base.py`

**❌ REMOVE These Failed Components**:
```python
class BasePipelineConfig(BaseModel):
    # REMOVE: Failed private field
    _portable_source_dir: Optional[str] = PrivateAttr(default=None)
    
    # REMOVE: Failed property
    @property
    def portable_source_dir(self) -> Optional[str]:
        """Get source directory as relative path for portability."""
        if self.source_dir is None:
            return None
            
        if self._portable_source_dir is None:
            self._portable_source_dir = self._convert_to_relative_path(self.source_dir)
        
        return self._portable_source_dir
    
    # REMOVE: Failed path conversion method
    def _convert_to_relative_path(self, path: str) -> str:
        """Convert absolute path to relative path based on runtime instantiation location."""
        # This entire method is fundamentally flawed and must be removed
        pass
    
    # REMOVE: Failed fallback method
    def _convert_via_common_parent(self, path: str) -> str:
        """Fallback conversion using common parent directory."""
        # This method also fails across deployment contexts
        pass
    
    # REMOVE: Failed helper method
    def _find_common_parent(self, path1: Path, path2: Path) -> Optional[Path]:
        """Find common parent directory of two paths."""
        # Helper method for failed approach
        pass
```

**✅ ADD New Multi-Strategy Implementation**:
```python
class BasePipelineConfig(BaseModel, ABC):
    """Base configuration with multi-strategy path resolution."""
    
    # Existing field - UNCHANGED (no breaking changes)
    source_dir: Optional[str] = Field(
        default=None,
        description="Source directory for scripts (relative path)"
    )
    
    # NEW: Project root folder for bundled deployments
    project_root_folder: Optional[str] = Field(
        default=None,
        description="Root folder name for user's project (e.g., 'mods_pipeline_adapter', 'fraud_detection')"
    )
    
    def _resolve_source_dir(self, project_root_folder: Optional[str], relative_path: Optional[str]) -> Optional[str]:
        """Universal path resolution for all deployment scenarios."""
        if not relative_path:
            return None
        
        scenario = self._detect_deployment_scenario()
        
        if scenario == "bundled":
            # Scenario 1: Lambda/MODS - use explicit project_root_folder
            package_root = Path(__file__).parent.parent  # cursus -> package_root
            if project_root_folder:
                # Direct path resolution using specified project folder
                resolved_path = package_root / project_root_folder / relative_path
                if resolved_path.exists():
                    return str(resolved_path)
            else:
                # Fallback: search through project folders (backward compatibility)
                return self._find_bundled_project_path(package_root, relative_path)
        
        elif scenario == "monorepo":
            # Scenario 2: Development - find project root via src/
            cursus_file = Path(__file__)
            src_index = cursus_file.parts.index("src")
            project_root = Path(*cursus_file.parts[:src_index])
            return str(project_root / relative_path)
        
        elif scenario == "pip_installed":
            # Scenario 3: Pip-installed - use working directory discovery
            return self._discover_user_project_path(relative_path)
        
        return None
    
    def _detect_deployment_scenario(self) -> str:
        """Detect which deployment scenario we're in."""
        cursus_file = Path(__file__)  # Current cursus module file
        
        # Strategy 1: Check if any project folders exist as siblings to cursus
        # (Scenario 1: Lambda/MODS bundled deployment)
        potential_package_root = cursus_file.parent.parent  # Go up from cursus/
        
        # Look for any sibling directories to cursus (excluding cursus itself)
        has_sibling_projects = False
        for item in potential_package_root.iterdir():
            if item.is_dir() and item.name != "cursus":
                has_sibling_projects = True
                break
        
        if has_sibling_projects:
            return "bundled"
        
        # Strategy 2: Check if we're in src/cursus structure
        # (Scenario 2: Development monorepo)
        if "src" in cursus_file.parts:
            src_index = cursus_file.parts.index("src")
            project_root = Path(*cursus_file.parts[:src_index])
            # Check if project root exists and is a valid directory
            if project_root.exists() and project_root.is_dir():
                return "monorepo"
        
        # Strategy 3: Pip-installed cursus, need to find user project
        # (Scenario 3: Separated pip installation)
        return "pip_installed"
    
    def _find_bundled_project_path(self, package_root: Path, relative_path: str) -> Optional[str]:
        """Fallback: Find project folder in bundled deployment (for backward compatibility)."""
        # Look through all sibling directories to cursus
        for item in package_root.iterdir():
            if item.is_dir() and item.name != "cursus":
                # Check if this project folder contains the relative_path
                potential_path = item / relative_path
                if potential_path.exists():
                    return str(potential_path)
        
        # Fallback: return None if no project folder contains the path
        return None
    
    def _discover_user_project_path(self, relative_path: str) -> Optional[str]:
        """Discover user project root when cursus is pip-installed."""
        # Start from current working directory
        current = Path.cwd()
        
        # Look for project markers
        markers = ['dockers/', 'pyproject.toml', '.git', 'config.json']
        
        # Search upward for project root
        while current != current.parent:
            if any((current / marker).exists() for marker in markers):
                # Found project root
                target_path = current / relative_path
                if target_path.exists():
                    return str(target_path)
            current = current.parent
        
        # Fallback: assume current directory is project root
        fallback_path = Path.cwd() / relative_path
        if fallback_path.exists():
            return str(fallback_path)
        
        return None
```

**Implementation Tasks**:
- [ ] Remove all `portable_source_dir` related code from `BasePipelineConfig`
- [ ] Remove `_convert_to_relative_path()` and related methods
- [ ] Add `project_root_folder` field to configuration
- [ ] Implement `_resolve_source_dir()` with multi-strategy approach
- [ ] Implement `_detect_deployment_scenario()` method
- [ ] Add helper methods for bundled and pip-installed scenarios
- [ ] Create comprehensive unit tests for scenario detection

#### **Day 3-4: Remove Processing-Specific Portable Path Code**

**Target File**: `src/cursus/steps/configs/config_processing_step_base.py`

**❌ REMOVE These Failed Components**:
```python
class ProcessingStepConfigBase(BasePipelineConfig):
    # REMOVE: Failed private field
    _portable_processing_source_dir: Optional[str] = PrivateAttr(default=None)
    _portable_script_path: Optional[str] = PrivateAttr(default=None)
    
    # REMOVE: Failed property
    @property
    def portable_processing_source_dir(self) -> Optional[str]:
        """Get processing source directory as relative path for portability."""
        # This property generates incorrect paths
        pass
    
    # REMOVE: Failed property
    @property
    def portable_effective_source_dir(self) -> Optional[str]:
        """Get effective source directory as relative path for step builders to use."""
        # This property also fails
        pass
    
    # REMOVE: Failed method
    def get_portable_script_path(self, default_path: Optional[str] = None) -> Optional[str]:
        """Get script path as relative path for portability."""
        # This method produces non-working paths
        pass
```

**✅ ADD New Multi-Strategy Implementation**:
```python
class ProcessingStepConfigBase(BasePipelineConfig):
    """Processing configuration with multi-strategy path resolution."""
    
    # Existing fields - UNCHANGED (no breaking changes)
    processing_source_dir: Optional[str] = Field(
        default=None,
        description="Source directory for processing scripts. Falls back to base source_dir if not provided."
    )
    
    @property
    def resolved_processing_source_dir(self) -> Optional[str]:
        """Get resolved processing source directory using multi-strategy approach."""
        if self.processing_source_dir:
            return self._resolve_source_dir(self.project_root_folder, self.processing_source_dir)
        return self._resolve_source_dir(self.project_root_folder, self.source_dir)
    
    def get_resolved_script_path(self) -> Optional[str]:
        """Get resolved script path for step builders using multi-strategy approach."""
        source_dir = self.resolved_processing_source_dir
        if source_dir and hasattr(self, 'script_name'):
            return str(Path(source_dir) / self.script_name)
        return None
```

**Implementation Tasks**:
- [ ] Remove all `portable_processing_source_dir` related code
- [ ] Remove `get_portable_script_path()` method
- [ ] Add `resolved_processing_source_dir` property using multi-strategy resolution
- [ ] Add `get_resolved_script_path()` method
- [ ] Update serialization to remove portable path fields
- [ ] Create unit tests for processing-specific multi-strategy resolution

#### **Day 5: Clean Up Configuration Serialization**

**Target**: Remove portable path fields from configuration serialization

**❌ REMOVE From Serialization**:
```json
{
  "source_dir": "/home/user/cursus/dockers/xgboost_atoz",
  "portable_source_dir": "../../dockers/xgboost_atoz",  // REMOVE
  "processing_source_dir": "/home/user/cursus/dockers/xgboost_atoz/scripts",
  "portable_processing_source_dir": "../../dockers/xgboost_atoz/scripts",  // REMOVE
  "portable_script_path": "../../dockers/xgboost_atoz/scripts/tabular_preprocessing.py"  // REMOVE
}
```

**✅ ADD Enhanced Serialization**:
```json
{
  "source_dir": "dockers/xgboost_atoz",
  "project_root_folder": "mods_pipeline_adapter",
  "processing_source_dir": "dockers/xgboost_atoz/scripts",
  "script_name": "tabular_preprocessing.py"
}
```

**Implementation Tasks**:
- [ ] Remove portable path fields from `model_dump()` methods
- [ ] Add `project_root_folder` to serialization
- [ ] Update configuration loading to handle new format
- [ ] Maintain backward compatibility with existing configs
- [ ] Add format validation and migration capabilities

#### **Phase 1 Success Criteria**
- [ ] All portable path infrastructure completely removed
- [ ] Multi-strategy path resolution implemented in base classes
- [ ] `project_root_folder` field added and functional
- [ ] Configuration serialization cleaned up
- [ ] Zero breaking changes - all existing code continues working
- [ ] Comprehensive unit test coverage for new implementation

### **Phase 2: Update Step Builders for Multi-Strategy Resolution** (Week 2)

#### **Objective**: Update all step builders to use the new multi-strategy resolution approach

#### **Day 1-2: Training Step Builders Enhancement**

**Target Files**: 
- `src/cursus/steps/builders/builder_xgboost_training_step.py`
- `src/cursus/steps/builders/builder_pytorch_training_step.py`

**❌ REMOVE Failed Portable Path Usage**:
```python
# XGBoost Training Step Builder - REMOVE
def _create_estimator(self, output_path=None) -> XGBoost:
    return XGBoost(
        entry_point=self.config.training_entry_point,
        source_dir=self.config.portable_source_dir or self.config.source_dir,  # ← REMOVE portable_source_dir
        framework_version=self.config.framework_version,
        # ... other parameters
    )
```

**✅ ADD Multi-Strategy Resolution**:
```python
# XGBoost Training Step Builder - ENHANCED
def _create_estimator(self, output_path=None) -> XGBoost:
    # Use multi-strategy resolution
    resolved_source_dir = self.config._resolve_source_dir(
        self.config.project_root_folder, 
        self.config.source_dir
    ) or self.config.source_dir  # Fallback to original
    
    return XGBoost(
        entry_point=self.config.training_entry_point,
        source_dir=resolved_source_dir,  # ← Multi-strategy resolution
        framework_version=self.config.framework_version,
        # ... other parameters
    )
```

**Implementation Tasks**:
- [ ] Remove `portable_source_dir` usage from XGBoostTrainingStepBuilder
- [ ] Remove `portable_source_dir` usage from PyTorchTrainingStepBuilder
- [ ] Add multi-strategy resolution calls with fallbacks
- [ ] Add logging to track resolution strategy used
- [ ] Create integration tests for training step creation
- [ ] Verify backward compatibility with existing training configurations

#### **Day 2-3: Model Step Builders Enhancement**

**Target Files**:
- `src/cursus/steps/builders/builder_xgboost_model_step.py`
- `src/cursus/steps/builders/builder_pytorch_model_step.py`

**❌ REMOVE Failed Portable Path Usage**:
```python
# XGBoost Model Step Builder - REMOVE
def _create_model(self, model_data: str) -> XGBoostModel:
    return XGBoostModel(
        model_data=model_data,
        entry_point=self.config.entry_point,
        source_dir=self.config.portable_source_dir or self.config.source_dir,  # ← REMOVE portable_source_dir
        framework_version=self.config.framework_version,
        # ... other parameters
    )
```

**✅ ADD Multi-Strategy Resolution**:
```python
# XGBoost Model Step Builder - ENHANCED
def _create_model(self, model_data: str) -> XGBoostModel:
    # Use multi-strategy resolution
    resolved_source_dir = self.config._resolve_source_dir(
        self.config.project_root_folder, 
        self.config.source_dir
    ) or self.config.source_dir  # Fallback to original
    
    return XGBoostModel(
        model_data=model_data,
        entry_point=self.config.entry_point,
        source_dir=resolved_source_dir,  # ← Multi-strategy resolution
        framework_version=self.config.framework_version,
        # ... other parameters
    )
```

**Implementation Tasks**:
- [ ] Remove `portable_source_dir` usage from XGBoostModelStepBuilder
- [ ] Remove `portable_source_dir` usage from PyTorchModelStepBuilder
- [ ] Add multi-strategy resolution calls with fallbacks
- [ ] Add logging to track resolution strategy used
- [ ] Create integration tests for model step creation
- [ ] Verify backward compatibility with existing model configurations

#### **Day 3-4: Processing Step Builders Enhancement**

**Target Files**:
- `src/cursus/steps/builders/builder_tabular_preprocessing_step.py`
- `src/cursus/steps/builders/builder_model_calibration_step.py`
- `src/cursus/steps/builders/builder_package_step.py`
- Additional processing step builders

**❌ REMOVE Failed Portable Path Usage**:
```python
# Tabular Preprocessing Step Builder - REMOVE
def build_step(self, **kwargs) -> ProcessingStep:
    # ... setup code ...
    script_path = self.config.get_portable_script_path() or self.config.get_script_path()  # ← REMOVE get_portable_script_path()
    
    step = ProcessingStep(
        name=step_name,
        processor=processor,
        code=script_path,  # Uses failed portable path
        # ... other parameters
    )
```

**✅ ADD Multi-Strategy Resolution**:
```python
# Tabular Preprocessing Step Builder - ENHANCED
def build_step(self, **kwargs) -> ProcessingStep:
    # ... setup code ...
    script_path = self.config.get_resolved_script_path() or self.config.get_script_path()  # ← Multi-strategy resolution
    
    step = ProcessingStep(
        name=step_name,
        processor=processor,
        code=script_path,  # Uses multi-strategy resolved path
        # ... other parameters
    )
```

**Implementation Tasks**:
- [ ] Remove `get_portable_script_path()` usage from TabularPreprocessingStepBuilder
- [ ] Remove `get_portable_script_path()` usage from ModelCalibrationStepBuilder
- [ ] Remove `get_portable_script_path()` usage from PackageStepBuilder
- [ ] Update remaining processing step builders (CurrencyConversion, PayloadStep, RiskTableMapping, DummyTraining, XGBoostModelEval)
- [ ] Add multi-strategy resolution calls with fallbacks
- [ ] Add logging to track resolution strategy used
- [ ] Create integration tests for processing step creation
- [ ] Verify backward compatibility with existing processing configurations

#### **Day 4-5: Remaining Step Builders Enhancement**

**Target Files**:
- `src/cursus/steps/builders/builder_cradle_data_loading_step.py`
- `src/cursus/steps/builders/builder_currency_conversion_step.py`
- `src/cursus/steps/builders/builder_risk_table_mapping_step.py`
- `src/cursus/steps/builders/builder_payload_step.py`
- `src/cursus/steps/builders/builder_xgboost_model_eval_step.py`
- `src/cursus/steps/builders/builder_registration_step.py`

**Implementation Strategy**: Apply the same pattern to all remaining step builders:
```python
# Pattern for Processing Step Builders
script_path = self.config.get_resolved_script_path() or self.config.get_script_path()

# Pattern for Training/Model Step Builders  
resolved_source_dir = self.config._resolve_source_dir(
    self.config.project_root_folder, 
    self.config.source_dir
) or self.config.source_dir
```

**Implementation Tasks**:
- [ ] Apply multi-strategy resolution pattern to all remaining step builders
- [ ] Remove all `portable_*` method usage
- [ ] Verify BatchTransformStepBuilder requires no changes (uses dependency resolution)
- [ ] Add comprehensive logging for path usage tracking
- [ ] Create integration tests for all updated step builders
- [ ] Validate complete backward compatibility across all step builders

#### **Phase 2 Success Criteria**
- [ ] All step builders updated to use multi-strategy resolution
- [ ] All `portable_*` method usage removed
- [ ] Comprehensive integration testing completed
- [ ] Zero breaking changes confirmed across all step builders
- [ ] Multi-strategy resolution logging implemented for monitoring

### **Phase 3: Testing and Validation** (Week 2-3)

#### **Objective**: Comprehensive testing across all deployment scenarios with realistic simulation

#### **Day 1-2: Multi-Strategy Resolution Unit Testing**

**Target Directory**: `test/core/multi_strategy_path_resolution/`

**New Test Files**:
- `test_deployment_scenario_detection.py`
- `test_bundled_deployment_resolution.py`
- `test_monorepo_deployment_resolution.py`
- `test_pip_installed_deployment_resolution.py`
- `test_step_builder_integration.py`

**Unit Test Coverage**:
```python
class TestDeploymentScenarioDetection:
    """Test automatic deployment scenario detection."""
    
    def test_bundled_scenario_detection(self):
        """Test detection of bundled deployment (Lambda/MODS)."""
        # Simulate Lambda structure with cursus and mods_pipeline_adapter as siblings
        with tempfile.TemporaryDirectory() as temp_dir:
            package_root = Path(temp_dir)
            cursus_dir = package_root / "cursus"
            project_dir = package_root / "mods_pipeline_adapter"
            
            cursus_dir.mkdir()
            project_dir.mkdir()
            
            # Mock cursus.__file__ to point to test structure
            with patch('cursus.__file__', str(cursus_dir / "__init__.py")):
                config = BasePipelineConfig()
                scenario = config._detect_deployment_scenario()
                assert scenario == "bundled"
    
    def test_monorepo_scenario_detection(self):
        """Test detection of monorepo deployment (development)."""
        # Simulate development structure with src/cursus
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            cursus_dir = project_root / "src" / "cursus"
            
            cursus_dir.mkdir(parents=True)
            
            # Mock cursus.__file__ to point to test structure
            with patch('cursus.__file__', str(cursus_dir / "__init__.py")):
                config = BasePipelineConfig()
                scenario = config._detect_deployment_scenario()
                assert scenario == "monorepo"
    
    def test_pip_installed_scenario_detection(self):
        """Test detection of pip-installed deployment."""
        # Simulate pip-installed structure
        with tempfile.TemporaryDirectory() as temp_dir:
            site_packages = Path(temp_dir) / "site-packages"
            cursus_dir = site_packages / "cursus"
            
            cursus_dir.mkdir(parents=True)
            
            # Mock cursus.__file__ to point to test structure
            with patch('cursus.__file__', str(cursus_dir / "__init__.py")):
                config = BasePipelineConfig()
                scenario = config._detect_deployment_scenario()
                assert scenario == "pip_installed"

class TestBundledDeploymentResolution:
    """Test path resolution for bundled deployment scenario."""
    
    def test_bundled_path_resolution_with_project_root_folder(self):
        """Test bundled path resolution using project_root_folder."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create Lambda-style structure
            package_root = Path(temp_dir)
            cursus_dir = package_root / "cursus"
            project_dir = package_root / "mods_pipeline_adapter"
            target_dir = project_dir / "dockers" / "xgboost_atoz"
            
            cursus_dir.mkdir()
            target_dir.mkdir(parents=True)
            
            config = BasePipelineConfig(
                project_root_folder="mods_pipeline_adapter",
                source_dir="dockers/xgboost_atoz"
            )
            
            with patch('cursus.__file__', str(cursus_dir / "__init__.py")):
                resolved = config._resolve_source_dir(
                    config.project_root_folder, 
                    config.source_dir
                )
                assert resolved == str(target_dir)
    
    def test_bundled_fallback_search(self):
        """Test bundled fallback when project_root_folder not specified."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create structure with multiple project folders
            package_root = Path(temp_dir)
            cursus_dir = package_root / "cursus"
            project1_dir = package_root / "project1"
            project2_dir = package_root / "project2"
            target_dir = project2_dir / "dockers" / "xgboost_atoz"
            
            cursus_dir.mkdir()
            project1_dir.mkdir()
            target_dir.mkdir(parents=True)
            
            config = BasePipelineConfig(source_dir="dockers/xgboost_atoz")
            
            with patch('cursus.__file__', str(cursus_dir / "__init__.py")):
                resolved = config._resolve_source_dir(None, config.source_dir)
                assert resolved == str(target_dir)

class TestMonorepoDeploymentResolution:
    """Test path resolution for monorepo deployment scenario."""
    
    def test_monorepo_path_resolution(self):
        """Test monorepo path resolution via src/ detection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create development structure
            project_root = Path(temp_dir)
            cursus_dir = project_root / "src" / "cursus"
            target_dir = project_root / "dockers" / "xgboost_atoz"
            
            cursus_dir.mkdir(parents=True)
            target_dir.mkdir(parents=True)
            
            config = BasePipelineConfig(source_dir="dockers/xgboost_atoz")
            
            with patch('cursus.__file__', str(cursus_dir / "__init__.py")):
                resolved = config._resolve_source_dir(None, config.source_dir)
                assert resolved == str(target_dir)

class TestPipInstalledDeploymentResolution:
    """Test path resolution for pip-installed deployment scenario."""
    
    def test_pip_installed_working_directory_discovery(self):
        """Test pip-installed path resolution via working directory discovery."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create pip-installed structure
            site_packages = Path(temp_dir) / "site-packages"
            cursus_dir = site_packages / "cursus"
            user_project = Path(temp_dir) / "user_project"
            target_dir = user_project / "dockers" / "xgboost_atoz"
            
            cursus_dir.mkdir(parents=True)
            target_dir.mkdir(parents=True)
            
            config = BasePipelineConfig(source_dir="dockers/xgboost_atoz")
            
            with patch('cursus.__file__', str(cursus_dir / "__init__.py")):
                with patch('pathlib.Path.cwd', return_value=user_project):
                    resolved = config._resolve_source_dir(None, config.source_dir)
                    assert resolved == str(target_dir)
```

**Implementation Tasks**:
- [ ] Create comprehensive unit test suite for all three deployment scenarios
- [ ] Test scenario detection logic with realistic file system structures
- [ ] Test path resolution for each scenario with various configurations
- [ ] Test fallback mechanisms and edge cases
- [ ] Add performance benchmarks for path resolution operations

#### **Day 3-4: Integration Testing**

**Target Directory**: `test/integration/multi_strategy_path_resolution/`

**Integration Test Scenarios**:
```python
class TestStepBuilderIntegration:
    """Test step builder integration with multi-strategy path resolution."""
    
    def test_xgboost_training_step_multi_strategy_resolution(self):
        """Test XGBoost training step creation with multi-strategy resolution."""
        config = XGBoostTrainingConfig(
            project_root_folder="mods_pipeline_adapter",
            source_dir="dockers/xgboost_atoz",
            training_entry_point="xgboost_training.py",
            # ... other required fields
        )
        
        builder = XGBoostTrainingStepBuilder(config)
        step = builder.create_step()
        
        # Verify step was created successfully with resolved paths
        assert step is not None
        assert step.name is not None
    
    def test_processing_step_multi_strategy_resolution(self):
        """Test processing step creation with multi-strategy resolution."""
        config = TabularPreprocessingConfig(
            project_root_folder="mods_pipeline_adapter",
            processing_source_dir="dockers/xgboost_atoz/scripts",
            processing_entry_point="tabular_preprocessing.py",
            # ... other required fields
        )
        
        builder = TabularPreprocessingStepBuilder(config)
        step = builder.create_step()
        
        # Verify step was created successfully with resolved script paths
        assert step is not None
        assert step.name is not None

class TestCrossEnvironmentCompatibility:
    """Test compatibility across different deployment environments."""
    
    def test_lambda_mods_environment_compatibility(self):
        """Test multi-strategy resolution works in Lambda/MODS environment."""
        # Simulate Lambda structure and test path resolution
        pass
    
    def test_development_environment_compatibility(self):
        """Test multi-strategy resolution works in development environment."""
        # Test with monorepo-style structure
        pass
    
    def test_pip_installed_environment_compatibility(self):
        """Test multi-strategy resolution works with pip-installed cursus."""
        # Test with separated cursus and user project
        pass
```

**Implementation Tasks**:
- [ ] Create integration tests for all step builder types
- [ ] Test cross-environment compatibility scenarios
- [ ] Verify end-to-end pipeline creation with multi-strategy configs
- [ ] Test configuration file loading and saving with new format
- [ ] Validate SageMaker step creation works correctly

#### **Day 5: Performance and Deployment Testing**

**Performance Testing**:
```python
class TestMultiStrategyPathResolutionPerformance:
    """Test performance impact of multi-strategy path resolution."""
    
    def test_scenario_detection_performance(self):
        """Benchmark scenario detection performance."""
        config = BasePipelineConfig()
        
        # Benchmark scenario detection
        start_time = time.time()
        for _ in range(1000):
            _ = config._detect_deployment_scenario()
        end_time = time.time()
        
        # Should be fast due to file system caching
        assert (end_time - start_time) < 0.5
    
    def test_path_resolution_performance(self):
        """Benchmark path resolution performance."""
        config = BasePipelineConfig(
            project_root_folder="mods_pipeline_adapter",
            source_dir="dockers/xgboost_atoz"
        )
        
        # Benchmark path resolution
        start_time = time.time()
        for _ in range(100):
            _ = config._resolve_source_dir(
                config.project_root_folder, 
                config.source_dir
            )
        end_time = time.time()
        
        # Should have minimal performance impact
        assert (end_time - start_time) < 1.0
```

**Deployment Testing**:
- [ ] Test in simulated Lambda/MODS environment
- [ ] Test in development monorepo environment
- [ ] Test in pip-installed package environment
- [ ] Test with different Python versions (3.8, 3.9, 3.10, 3.11)
- [ ] Validate memory usage and performance impact

#### **Phase 3 Success Criteria**
- [ ] Comprehensive unit test coverage (>95%) for all multi-strategy functionality
- [ ] Integration tests pass for all step builder types
- [ ] Cross-environment compatibility validated
- [ ] Performance impact minimal (<5% overhead)
- [ ] Deployment testing successful across all target environments

### **Phase 4: Documentation and Finalization** (Week 3)

#### **Objective**: Complete documentation, migration guides, and production readiness

#### **Day 1-2: Developer Documentation**

**Target Files**:
- `docs/multi_strategy_path_resolution_guide.md`
- `docs/deployment_scenario_compatibility.md`
- `docs/configuration_field_reference.md`

**Documentation Content**:
```markdown
# Multi-Strategy Deployment Path Resolution Guide

## Overview
The Multi-Strategy Deployment Path Resolution System automatically detects deployment scenarios and applies the appropriate path resolution strategy, enabling cursus configurations to work seamlessly across all deployment environments.

## Key Features
- **Automatic Scenario Detection**: System detects bundled, monorepo, and pip-installed deployments
- **Universal Compatibility**: Same config files work across Lambda/MODS, development, and pip-installed environments
- **Zero Breaking Changes**: All existing code continues working without modification
- **Transparent Operation**: Users see no difference in behavior

## Configuration Fields

### New Fields
- **`project_root_folder`**: Root folder name for bundled deployments (e.g., "mods_pipeline_adapter")

### Enhanced Fields
- **`source_dir`**: Now works universally across all deployment scenarios
- **`processing_source_dir`**: Enhanced with multi-strategy resolution

## Usage Examples

### Bundled Deployment (Lambda/MODS)
```python
config = XGBoostTrainingConfig(
    project_root_folder="mods_pipeline_adapter",  # Specifies project folder
    source_dir="dockers/xgboost_atoz",
    training_entry_point="xgboost_training.py"
)
```

### Monorepo Development
```python
config = XGBoostTrainingConfig(
    source_dir="dockers/xgboost_atoz",  # project_root_folder ignored
    training_entry_point="xgboost_training.py"
)
```

### Pip-Installed Separated
```python
config = XGBoostTrainingConfig(
    source_dir="dockers/xgboost_atoz",  # project_root_folder ignored
    training_entry_point="xgboost_training.py"
)
```
```

**Implementation Tasks**:
- [ ] Create comprehensive developer documentation
- [ ] Document all new configuration fields and their usage
- [ ] Provide usage examples for different deployment scenarios
- [ ] Create troubleshooting guide for path resolution issues
- [ ] Document migration from previous portable path approach

#### **Day 2-3: Migration Guide**

**Target File**: `docs/multi_strategy_path_resolution_migration_guide.md`

**Migration Guide Content**:
```markdown
# Multi-Strategy Path Resolution Migration Guide

## Migration Overview
The Multi-Strategy Path Resolution System replaces the previous portable path approach with a more robust, deployment-aware solution. This migration **removes failed portable path infrastructure** and implements universal deployment compatibility.

## What Changed
- **Removed**: `portable_source_dir`, `portable_processing_source_dir` properties and related methods
- **Added**: `project_root_folder` field for bundled deployments
- **Enhanced**: Multi-strategy path resolution with automatic scenario detection

## What Didn't Change
- **Existing APIs**: All existing methods and properties work exactly as before
- **Configuration Format**: Original fields preserved, only portable path fields removed
- **User Workflow**: No changes required to existing development practices

## Migration Steps

### Step 1: Update Cursus (Automatic)
```bash
pip install --upgrade cursus
```

### Step 2: Update Configurations (Optional)
For bundled deployments (Lambda/MODS), add `project_root_folder`:
```json
{
  "project_root_folder": "mods_pipeline_adapter",
  "source_dir": "dockers/xgboost_atoz"
}
```

### Step 3: Remove Portable Path Usage (If Any)
If you were using portable path properties directly:
```python
# REMOVE: Direct portable path usage
script_path = config.get_portable_script_path()

# REPLACE: With multi-strategy resolution
script_path = config.get_resolved_script_path()
```

## Benefits After Migration
- Universal deployment compatibility
- Automatic scenario detection
- Robust fallback mechanisms
- Enhanced error handling
```

**Implementation Tasks**:
- [ ] Create migration guide from portable path approach
- [ ] Document configuration changes required
- [ ] Provide code migration examples
- [ ] Create troubleshooting procedures
- [ ] Add FAQ section for common migration questions

#### **Day 3-4: Production Readiness**

**Production Checklist**:
- [ ] **Code Quality**: All code reviewed and approved
- [ ] **Test Coverage**: >95% test coverage achieved
- [ ] **Performance**: <5% performance impact verified
- [ ] **Documentation**: Complete documentation available
- [ ] **Backward Compatibility**: 100% compatibility confirmed
- [ ] **Deployment Testing**: All environments tested successfully

**Monitoring and Logging**:
```python
# Enhanced logging for production monitoring
logger.info(f"Detected deployment scenario: {scenario}")
logger.info(f"Path resolution: {relative_path} -> {resolved_path}")
logger.warning(f"Path resolution failed, using fallback: {relative_path}")
logger.error(f"Multi-strategy resolution error: {error_message}")
```

**Error Handling Enhancement**:
```python
def _resolve_source_dir(self, project_root_folder: Optional[str], relative_path: Optional[str]) -> Optional[str]:
    """Enhanced error handling for production."""
    try:
        # Primary multi-strategy resolution
        return self._primary_multi_strategy_resolution(project_root_folder, relative_path)
    except Exception as e:
        logger.warning(f"Multi-strategy resolution failed: {e}")
        try:
            # Fallback to original path
            return relative_path
        except Exception as e2:
            logger.error(f"All path resolution strategies failed: {e2}")
            return None
```

**Implementation Tasks**:
- [ ] Enhance error handling and logging for production
- [ ] Add monitoring capabilities for path resolution success/failure rates
- [ ] Create production deployment checklist
- [ ] Implement graceful degradation strategies
- [ ] Add performance monitoring and alerting

#### **Day 4-5: Final Testing and Release Preparation**

**Final Testing Checklist**:
- [ ] **Unit Tests**: All unit tests passing (>95% coverage)
- [ ] **Integration Tests**: All integration tests passing
- [ ] **Performance Tests**: Performance impact <5%
- [ ] **Compatibility Tests**: Backward compatibility 100%
- [ ] **Deployment Tests**: All environments tested successfully
- [ ] **Documentation Tests**: All examples work correctly

**Release Preparation**:
- [ ] **Version Bump**: Update version numbers appropriately
- [ ] **Changelog**: Document all changes and improvements
- [ ] **Release Notes**: Create user-friendly release notes
- [ ] **Deployment Plan**: Finalize production deployment strategy
- [ ] **Rollback Plan**: Prepare rollback procedures if needed

#### **Phase 4 Success Criteria**
- [ ] Complete documentation available for developers
- [ ] Migration guide from portable path approach published
- [ ] Production readiness checklist completed
- [ ] Final testing successful across all scenarios
- [ ] Release preparation completed

## Risk Management

### **High Risk Items**

#### **Risk 1: Scenario Detection Failures in Edge Cases**
- **Probability**: Medium
- **Impact**: High
- **Mitigation**: 
  - Implement comprehensive fallback mechanisms
  - Test across diverse deployment environments
  - Add detailed logging for scenario detection
  - Provide manual override capabilities

#### **Risk 2: Performance Impact on Large-Scale Deployments**
- **Probability**: Low
- **Impact**: Medium
- **Mitigation**:
  - Implement caching for scenario detection
  - Performance benchmarking and optimization
  - Monitor path resolution performance in production
  - Optimize file system access patterns

#### **Risk 3: Backward Compatibility Issues with Existing Configs**
- **Probability**: Low
- **Impact**: High
- **Mitigation**:
  - Preserve all existing APIs and behaviors
  - Comprehensive compatibility testing
  - Gradual rollout with monitoring
  - Rollback plan with legacy implementation

### **Medium Risk Items**

#### **Risk 4: Complex Lambda/MODS Deployment Environment Issues**
- **Probability**: Medium
- **Impact**: Medium
- **Mitigation**:
  - Test in realistic Lambda simulation environments
  - Implement Lambda-specific fallbacks
  - Create Lambda deployment-specific documentation
  - Monitor Lambda deployment success rates

#### **Risk 5: Step Builder Integration Complexity**
- **Probability**: Low
- **Impact**: Medium
- **Mitigation**:
  - Use consistent pattern across all step builders
  - Comprehensive integration testing
  - Gradual rollout with monitoring
  - Clear rollback procedures for each step builder

## Success Metrics

### **Immediate Success Metrics** (Week 1)
- **Portable Path Infrastructure Removal**: ✅ All failed portable path code removed
- **Multi-Strategy Implementation**: ✅ Core multi-strategy resolution implemented
- **Configuration Enhancement**: ✅ `project_root_folder` field added and functional
- **Backward Compatibility**: ✅ 100% compatibility with existing code

### **Intermediate Success Metrics** (Week 2)
- **Step Builder Integration**: All step builders updated with multi-strategy resolution
- **Testing Coverage**: >95% test coverage for multi-strategy functionality
- **Cross-Environment Testing**: Successful testing across all deployment environments
- **Performance Impact**: <5% performance overhead confirmed

### **Final Success Metrics** (Week 3)
- **Universal Deployment Compatibility**: Configuration files work across all environments
- **Lambda/MODS Error Resolution**: Critical MODS pipeline deployment failures fixed
- **Developer Experience**: Zero-impact migration with enhanced capabilities
- **Production Readiness**: Complete documentation and monitoring capabilities

### **Long-term Success Metrics**
- **Configuration Sharing**: Developers can share configs seamlessly across environments
- **Deployment Success**: 100% success rate across all deployment environments
- **Maintenance Reduction**: Eliminated manual path adjustment requirements
- **System Reliability**: Robust fallback mechanisms and error handling

## Dependencies and Prerequisites

### **Required Dependencies**
- **Python 3.8+**: Required for pathlib and type hints
- **Pydantic**: For config class enhancement and serialization
- **Existing Config Architecture**: Three-tier config system must be operational

### **Development Environment**
- **Testing Framework**: pytest with comprehensive test coverage
- **Development Tools**: Code coverage, performance profiling, linting
- **Documentation Tools**: Markdown documentation with examples

### **Deployment Environments**
- **Local Development**: Standard Python development environment
- **Lambda/MODS**: AWS Lambda deployment testing environment
- **Container Environments**: Docker containerized deployment testing
- **PyPI Package**: Package distribution testing environment

## Quality Assurance

### **Code Quality Standards**
- **Test Coverage**: Minimum 95% line coverage for all new functionality
- **Documentation**: Comprehensive docstrings and user documentation
- **Type Hints**: Full type annotation for all public APIs
- **Code Style**: Consistent formatting and linting compliance

### **Testing Strategy**
- **Unit Tests**: Individual component functionality and edge cases
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
- **Critical Path Resolution Failures**: >10% path resolution failure rate
- **Performance Regression**: >10% performance degradation
- **Backward Compatibility Issues**: Any breaking changes detected
- **Deployment Failures**: Universal deployment compatibility lost

### **Rollback Procedure**
1. **Immediate Rollback**: Revert to previous working version
2. **Issue Analysis**: Identify root cause of failure
3. **Fix Implementation**: Address issues in development environment
4. **Gradual Re-deployment**: Phased re-introduction with enhanced monitoring

### **Rollback Assets**
- **Version Control**: Complete change history and rollback points
- **Legacy Compatibility**: Existing functionality preserved during implementation
- **Test Suites**: Comprehensive validation for rollback verification
- **Documentation**: Rollback procedures and troubleshooting guides

## Lessons Learned from Previous Implementation

### **Critical Insights from 2025-09-20 Plan Failure**

#### **❌ Single-Strategy Approach Limitations**
The previous portable path implementation failed because it attempted to use a single path conversion strategy across all deployment contexts. This approach was fundamentally flawed because:

1. **Context Dependency**: Path conversion relied on development-time context that didn't exist at deployment time
2. **Working Directory Assumptions**: Assumed consistent working directory across environments
3. **Lambda Architecture Mismatch**: Failed to account for Lambda's sibling directory structure

#### **✅ Multi-Strategy Approach Benefits**
The new multi-strategy approach addresses these failures by:

1. **Deployment-Aware Resolution**: Different strategies for different deployment contexts
2. **Automatic Detection**: System automatically detects which strategy to use
3. **Robust Fallbacks**: Multiple fallback mechanisms for edge cases
4. **Universal Compatibility**: Works across all deployment architectures

### **Key Implementation Insights**

1. **Deployment Context Matters**: Always consider where the code will be executed, not where it's defined
2. **File System Structure Varies**: Different deployment contexts have fundamentally different directory structures
3. **Test Realistic Scenarios**: Must simulate actual deployment contexts, not just development convenience
4. **Fallback Mechanisms Essential**: Multiple strategies needed for robust operation
5. **Backward Compatibility Critical**: Existing functionality must be preserved during enhancement

## Conclusion

This implementation plan provides a comprehensive roadmap for implementing the Multi-Strategy Deployment Path Resolution System to achieve universal deployment portability while maintaining complete backward compatibility. The phased approach ensures minimal risk while delivering immediate value through enhanced configuration sharing capabilities.

**Key Innovation**: Using **deployment scenario detection** with **strategy-specific path resolution** ensures compatibility across all deployment environments while avoiding the complexity and failures of the previous single-strategy approach.

**Final Deliverables**:
- **Universal Configuration Portability**: Same config files work across all environments
- **Zero Breaking Changes**: All existing code continues working without modification
- **Enhanced Developer Experience**: Transparent operation with improved capabilities
- **Production-Ready System**: Robust error handling, monitoring, and documentation
- **Lambda/MODS Error Resolution**: Fixes critical MODS pipeline deployment failures

## References

### **Design Documents**
- **[Multi-Strategy Deployment Path Resolution Design](../1_design/multi_strategy_deployment_path_resolution_design.md)** - Complete architectural design and technical specifications
- **[Deployment-Context-Agnostic Path Resolution Design](../1_design/deployment_context_agnostic_path_resolution_design.md)** - Previous approach analysis and lessons learned

### **Previous Implementation Plans**
- **[2025-09-20 Config Portability Path Resolution Implementation Plan](./2025-09-20_config_portability_path_resolution_implementation_plan.md)** - Previous portable path approach that failed and needs to be reverted
- **[2025-09-22 MODS Lambda Sibling Directory Path Resolution Fix Completion](./2025-09-22_mods_lambda_sibling_directory_path_resolution_fix_completion.md)** - Lambda-specific fixes that informed this design

### **Analysis Documents**
- **[MODS Pipeline Path Resolution Error Analysis](../.internal/mods_pipeline_path_resolution_error_analysis.md)** - Comprehensive error analysis that motivated this implementation

### **Configuration System Documentation**
- **[Three-Tier Config Design](../1_design/config_tiered_design.md)** - Configuration architecture patterns
- **[Config Field Categorization Consolidated](../1_design/config_field_categorization_consolidated.md)** - Field management principles

### **Supporting Documents**
- **[Cursus Package Portability Architecture Design](../1_design/cursus_package_portability_architecture_design.md)** - Overall portability architecture and deployment strategies
- **[Unified Step Catalog System Design](../1_design/unified_step_catalog_system_design.md)** - Step catalog integration architecture

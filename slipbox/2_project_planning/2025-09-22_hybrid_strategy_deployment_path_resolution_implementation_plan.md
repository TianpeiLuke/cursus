---
tags:
  - implementation_plan
  - hybrid_strategy
  - path_resolution
  - deployment_portability
keywords:
  - hybrid path resolution
  - package location discovery
  - working directory discovery
  - universal configuration
  - deployment scenarios
topics:
  - hybrid strategy implementation
  - path resolution algorithm
  - configuration portability
  - deployment universality
language: python
date of note: 2025-09-22
---

# Hybrid Strategy Deployment Path Resolution Implementation Plan

## Executive Summary

This implementation plan provides a comprehensive roadmap for implementing the **Hybrid Strategy Deployment Path Resolution** system in cursus. The solution addresses the fundamental challenge of configuration portability across three deployment scenarios: **Lambda/MODS bundled**, **development monorepo**, and **pip-installed separated environments**.

The hybrid approach implements **Package Location First** with **Working Directory Discovery** as fallback, eliminating the need for complex scenario detection while providing universal configuration portability.

## Problem Context

### Current Path Resolution Limitations

Based on analysis in `hybrid_strategy_deployment_path_resolution_design.md`, the current system fails because:

1. **Single deployment pattern assumption** - existing resolution assumes working directory connectivity
2. **Configuration fragility** - same config fails across different deployment contexts
3. **Manual path adjustments** - users must modify configurations for different environments
4. **Lambda/MODS failures** - complete filesystem separation breaks current resolution

### Solution Architecture: Hybrid Resolution Algorithm

The hybrid approach uses two complementary strategies:

#### **Strategy 1: Package Location Discovery (Primary)**
- Uses `Path(__file__)` from cursus package location as reference point
- Works across all deployment scenarios (bundled, monorepo, pip-installed)
- Handles monorepo structure detection (`src/cursus` patterns)
- Most reliable strategy - tried first

#### **Strategy 2: Working Directory Discovery (Fallback)**
- Uses `Path.cwd()` for working directory traversal
- Handles edge cases when package location fails
- Supports project root folder discovery
- Automatic fallback when Strategy 1 fails

## Implementation Phases

### Phase 0: Legacy Portable Path Cleanup (Week 0 - Preparation)

Before implementing the hybrid strategy, we need to remove the complex portable path infrastructure from the `2025-09-20_config_portability_path_resolution_implementation_plan.md` that is no longer needed.

#### **0.1 Remove Portable Path Fields**

**Files to Update**: All configuration classes in `src/cursus/steps/configs/`

**Remove these fields that are no longer needed:**
```python
# REMOVE: These portable path fields are no longer needed
class ProcessingStepConfigBase(BasePipelineConfig):
    # Remove these fields:
    # portable_source_dir: Optional[str] = Field(default=None, description="Portable source directory path")
    # portable_processing_source_dir: Optional[str] = Field(default=None, description="Portable processing source directory path")
    
    # Keep existing fields:
    processing_source_dir: Optional[str] = Field(
        default=None,
        description="Source directory for processing scripts (relative path)"
    )
```

#### **0.2 Remove Portable Path Methods**

**Files to Update**: `src/cursus/core/base/config_base.py` and related config classes

**Remove these methods that are no longer needed:**
```python
# REMOVE: These portable path methods are no longer needed
# def get_portable_source_dir(self) -> Optional[str]:
# def get_portable_processing_source_dir(self) -> Optional[str]:
# def get_portable_script_path(self) -> Optional[str]:
# def _convert_to_portable_path(self, absolute_path: str) -> str:
# def _resolve_portable_path(self, portable_path: str) -> str:
```

#### **0.3 Simplify Configuration Serialization**

**File**: `src/cursus/core/config_fields/config_merger.py`

**Remove portable path serialization logic:**
```python
# REMOVE: Complex portable path serialization
# No longer need to convert absolute paths to portable paths during serialization
# No longer need to resolve portable paths during deserialization

# KEEP: Simple field serialization for Tier 1 fields
# - project_root_folder: Direct serialization (just a folder name)
# - source_dir: Direct serialization (relative path)
```

#### **0.4 Update Step Builders to Remove Portable Path Usage**

**Files to Update**: All step builders in `src/cursus/steps/builders/`

Based on code analysis, the following step builders need portable path cleanup:

**Step builders using `get_portable_script_path()` and `get_resolved_path()`:**
- `builder_tabular_preprocessing_step.py`
- `builder_model_calibration_step.py`
- `builder_currency_conversion_step.py`
- `builder_payload_step.py`
- `builder_package_step.py`

**Step builders using `portable_source_dir` and `get_resolved_path()`:**
- `builder_pytorch_model_step.py`
- `builder_pytorch_training_step.py`
- `builder_xgboost_training_step.py`
- `builder_xgboost_model_step.py`
- `builder_package_step.py`

**Step builders using `portable_processing_source_dir` and `get_resolved_path()`:**
- `builder_xgboost_model_eval_step.py`

**Remove portable path usage pattern:**
```python
# BEFORE: Complex portable path resolution
# Get script path from config - use portable path with fallback
portable_script_path = self.config.get_portable_script_path()
if portable_script_path:
    # Resolve portable path to absolute path for SageMaker using runtime detection
    script_path = self.config.get_resolved_path(portable_script_path)
    self.log_info("Resolved portable path %s to %s", portable_script_path, script_path)
else:
    script_path = self.config.get_script_path()
    self.log_info("Using script path: %s (portable: no)", script_path)

# AFTER: Simple direct path usage (temporary - will be replaced with hybrid resolution in Phase 2)
script_path = self.config.get_script_path()
self.log_info("Using script path: %s", script_path)
```

**Remove portable source directory usage pattern:**
```python
# BEFORE: Complex portable source directory resolution
# Use portable path with fallback for universal deployment compatibility
portable_source_dir = self.config.portable_source_dir
if portable_source_dir:
    # Resolve portable path to absolute path for SageMaker using runtime detection
    source_dir = self.config.get_resolved_path(portable_source_dir)
    self.log_info("Resolved portable source dir %s to %s", portable_source_dir, source_dir)
else:
    source_dir = self.config.source_dir
    self.log_info("Using source dir: %s (portable: no)", source_dir)

# AFTER: Simple direct source directory usage (temporary - will be replaced with hybrid resolution in Phase 2)
source_dir = self.config.source_dir
self.log_info("Using source dir: %s", source_dir)
```

**Remove portable processing source directory usage pattern:**
```python
# BEFORE: Complex portable processing source directory resolution
# Use portable path with fallback for universal deployment compatibility
portable_processing_source_dir = self.config.portable_processing_source_dir
if portable_processing_source_dir:
    # Resolve portable path to absolute path for SageMaker using runtime detection
    source_dir = self.config.get_resolved_path(portable_processing_source_dir)
    self.log_info("Resolved portable processing source dir %s to %s", portable_processing_source_dir, source_dir)
else:
    source_dir = self.config.processing_source_dir or self.config.source_dir
    self.log_info("Using processing source dir: %s (portable: no)", source_dir)

# AFTER: Simple direct processing source directory usage (temporary - will be replaced with hybrid resolution in Phase 2)
source_dir = self.config.processing_source_dir or self.config.source_dir
self.log_info("Using processing source dir: %s", source_dir)
```

#### **0.5 Remove Portable Path Tests**

**Files to Update**: Test files related to portable path functionality

**Remove these test files/methods:**
```python
# REMOVE: These test files are no longer needed
# test/core/test_portable_path_resolution.py
# test/integration/test_portable_path_integration.py

# REMOVE: These test methods from existing test files
# def test_portable_source_dir_generation(self):
# def test_portable_path_resolution(self):
# def test_portable_script_path_creation(self):
```

#### **0.6 Update Configuration Examples**

**Files**: Documentation and example configuration files

**Manual Update**: Configuration examples can be updated manually as needed during development and documentation updates. No specific implementation required for Phase 0.

### Phase 1: Core Hybrid Algorithm Implementation (Week 1)

#### **1.1 Enhanced BasePipelineConfig**

**File**: `src/cursus/core/base/config_base.py`

```python
class BasePipelineConfig(BaseModel, ABC):
    """Base configuration with hybrid path resolution."""
    
    # Tier 1 required user input fields for universal configuration
    project_root_folder: str = Field(
        description="Root folder name for the user's project (Tier 1 required user input)"
    )
    source_dir: str = Field(
        description="Source directory for scripts relative to project_root_folder (Tier 1 required user input)"
    )
    
    def _resolve_source_dir(self, project_root_folder: str, relative_path: str) -> Optional[str]:
        """Hybrid path resolution: Package location first, then working directory discovery."""
        if not relative_path:
            return None
        
        # Strategy 1: Package Location Discovery (works for all scenarios)
        resolved = self._package_location_discovery(project_root_folder, relative_path)
        if resolved:
            return resolved
        
        # Strategy 2: Working Directory Discovery (fallback for edge cases)
        resolved = self._working_directory_discovery(project_root_folder, relative_path)
        if resolved:
            return resolved
        
        return None
    
    def _package_location_discovery(self, project_root_folder: Optional[str], relative_path: str) -> Optional[str]:
        """Discover paths using cursus package location as reference."""
        cursus_file = Path(__file__)  # Current cursus module file
        
        # Strategy 1A: Check for bundled deployment (Lambda/MODS)
        # Look for sibling directories to cursus
        potential_package_root = cursus_file.parent.parent  # Go up from cursus/
        
        # If project_root_folder is specified, use it directly
        if project_root_folder:
            direct_path = potential_package_root / project_root_folder / relative_path
            if direct_path.exists():
                return str(direct_path)
        
        # Try direct resolution from package root (for backward compatibility)
        direct_path = potential_package_root / relative_path
        if direct_path.exists():
            return str(direct_path)
        
        # Strategy 1B: Check if we're in monorepo structure (src/cursus)
        if "src" in cursus_file.parts:
            src_index = cursus_file.parts.index("src")
            project_root = Path(*cursus_file.parts[:src_index])
            
            if project_root.exists() and project_root.is_dir():
                if project_root_folder:
                    target_path = project_root / project_root_folder / relative_path
                else:
                    target_path = project_root / relative_path
                    
                if target_path.exists():
                    return str(target_path)
        
        return None
    
    def _working_directory_discovery(self, project_root_folder: Optional[str], relative_path: str) -> Optional[str]:
        """Discover paths using working directory traversal (fallback)."""
        current = Path.cwd()
        
        # Search upward for project root
        while current != current.parent:
            # Strategy 2A: If project_root_folder is specified, check if we're inside it
            if project_root_folder:
                # Check if current directory name matches project_root_folder
                if current.name == project_root_folder:
                    target_path = current / relative_path
                    if target_path.exists():
                        return str(target_path)
                
                # Check if project_root_folder exists as subdirectory of current
                project_folder_path = current / project_root_folder
                if project_folder_path.exists() and project_folder_path.is_dir():
                    target_path = project_folder_path / relative_path
                    if target_path.exists():
                        return str(target_path)
            
            # Strategy 2B: Direct path resolution (for cases without project_root_folder)
            direct_path = current / relative_path
            if direct_path.exists():
                return str(direct_path)
                
            current = current.parent
        
        # Final fallback: try current working directory
        if project_root_folder:
            fallback_with_project = Path.cwd() / project_root_folder / relative_path
            if fallback_with_project.exists():
                return str(fallback_with_project)
        
        fallback_path = Path.cwd() / relative_path
        if fallback_path.exists():
            return str(fallback_path)
        
        return None
    
    @property
    def resolved_source_dir(self) -> Optional[str]:
        """Get resolved source directory using hybrid resolution."""
        return self._resolve_source_dir(self.project_root_folder, self.source_dir)
```

#### **1.2 Enhanced ProcessingStepConfigBase**

**File**: `src/cursus/steps/configs/config_processing_step_base.py`

```python
class ProcessingStepConfigBase(BasePipelineConfig):
    """Processing configuration with hybrid path resolution."""
    
    # Existing fields unchanged
    processing_source_dir: Optional[str] = Field(
        default=None,
        description="Source directory for processing scripts (relative path)"
    )
    
    @property
    def resolved_processing_source_dir(self) -> Optional[str]:
        """Get resolved processing source directory using hybrid resolution."""
        if self.processing_source_dir:
            return self._resolve_source_dir(self.project_root_folder, self.processing_source_dir)
        return self._resolve_source_dir(self.project_root_folder, self.source_dir)
    
    def get_resolved_script_path(self) -> Optional[str]:
        """Get resolved script path for step builders."""
        source_dir = self.resolved_processing_source_dir
        if source_dir and hasattr(self, 'script_name'):
            return str(Path(source_dir) / self.script_name)
        return None
```

#### **1.3 Core Algorithm Testing**

**File**: `test/core/test_hybrid_path_resolution.py`

```python
class TestHybridPathResolution(unittest.TestCase):
    
    def test_bundled_deployment_resolution(self):
        """Test hybrid resolution for bundled deployment (Lambda/MODS)."""
        # Simulate Lambda/MODS structure
        with tempfile.TemporaryDirectory() as temp_dir:
            package_root = Path(temp_dir)
            cursus_dir = package_root / "cursus" / "core" / "base"
            project_dir = package_root / "mods_pipeline_adapter" / "dockers" / "xgboost_atoz"
            
            cursus_dir.mkdir(parents=True)
            project_dir.mkdir(parents=True)
            
            # Create test config
            config = TestConfig(
                project_root_folder="mods_pipeline_adapter",
                source_dir="dockers/xgboost_atoz",
                script_name="test_script.py"
            )
            
            # Test hybrid resolution
            with patch('pathlib.Path.__file__', str(cursus_dir / "config_base.py")):
                resolved = config._resolve_source_dir("mods_pipeline_adapter", "dockers/xgboost_atoz")
                self.assertEqual(resolved, str(project_dir))
    
    def test_monorepo_deployment_resolution(self):
        """Test hybrid resolution for monorepo deployment."""
        # Simulate development monorepo structure
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            cursus_dir = project_root / "src" / "cursus" / "core" / "base"
            project_dir = project_root / "project_xgboost_atoz"
            
            cursus_dir.mkdir(parents=True)
            project_dir.mkdir(parents=True)
            
            # Create test config
            config = TestConfig(
                project_root_folder="project_xgboost_atoz",
                source_dir=".",
                script_name="xgboost_training.py"
            )
            
            # Test hybrid resolution (monorepo detection)
            with patch('pathlib.Path.__file__', str(cursus_dir / "config_base.py")):
                resolved = config._resolve_source_dir("project_xgboost_atoz", ".")
                self.assertEqual(resolved, str(project_dir))
    
    def test_pip_installed_deployment_resolution(self):
        """Test hybrid resolution for pip-installed deployment."""
        # Simulate pip-installed structure
        with tempfile.TemporaryDirectory() as temp_dir:
            # System-wide cursus location
            site_packages = Path(temp_dir) / "usr" / "local" / "lib" / "python3.x" / "site-packages"
            cursus_dir = site_packages / "cursus" / "core" / "base"
            cursus_dir.mkdir(parents=True)
            
            # User project location
            user_project = Path(temp_dir) / "home" / "user" / "my_project"
            project_dir = user_project / "dockers" / "xgboost_atoz"
            project_dir.mkdir(parents=True)
            
            # Create test config
            config = TestConfig(
                project_root_folder="my_project",
                source_dir="dockers/xgboost_atoz",
                script_name="preprocessing.py"
            )
            
            # Test hybrid resolution (working directory discovery)
            with patch('pathlib.Path.__file__', str(cursus_dir / "config_base.py")), \
                 patch('pathlib.Path.cwd', return_value=user_project):
                
                resolved = config._resolve_source_dir("my_project", "dockers/xgboost_atoz")
                self.assertEqual(resolved, str(project_dir))
```

### Phase 2: Step Builder Integration (Week 2)

#### **2.1 Update Step Builders**

**Files**: All step builders in `src/cursus/steps/builders/`

**Pattern**: Update all step builders to use hybrid resolution:

```python
# BEFORE: Direct path usage
class TabularPreprocessingStepBuilder(StepBuilderBase):
    def create_step(self, **kwargs) -> ProcessingStep:
        script_path = self.config.get_script_path()  # May fail in different contexts
        
        return ProcessingStep(
            name=step_name,
            processor=processor,
            inputs=proc_inputs,
            outputs=proc_outputs,
            code=script_path,
            job_arguments=job_args,
            depends_on=dependencies,
            cache_config=self._get_cache_config(enable_caching),
        )

# AFTER: Hybrid resolution with fallback
class TabularPreprocessingStepBuilder(StepBuilderBase):
    def create_step(self, **kwargs) -> ProcessingStep:
        # Get script path using hybrid resolution with fallback
        script_path = (
            self.config.get_resolved_script_path() or  # Hybrid resolution
            self.config.get_script_path()              # Fallback to existing behavior
        )
        self.log_info("Using script path: %s", script_path)
        
        return ProcessingStep(
            name=step_name,
            processor=processor,
            inputs=proc_inputs,
            outputs=proc_outputs,
            code=script_path,  # Uses hybrid-resolved path
            job_arguments=job_args,
            depends_on=dependencies,
            cache_config=self._get_cache_config(enable_caching),
        )
```

#### **2.2 Step Builder Integration Testing**

**File**: `test/steps/builders/test_hybrid_integration.py`

```python
class TestStepBuilderHybridIntegration(unittest.TestCase):
    
    def test_tabular_preprocessing_hybrid_resolution(self):
        """Test TabularPreprocessingStepBuilder with hybrid resolution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create project structure
            project_root = Path(temp_dir)
            scripts_dir = project_root / "project_xgboost_pda" / "materials"
            scripts_dir.mkdir(parents=True)
            script_file = scripts_dir / "tabular_preprocessing.py"
            script_file.write_text("# Mock preprocessing script")
            
            # Create cursus structure for monorepo detection
            cursus_dir = project_root / "src" / "cursus" / "core" / "base"
            cursus_dir.mkdir(parents=True)
            
            # Create config with hybrid resolution fields
            config = TabularPreprocessingConfig(
                bucket="test-bucket",
                current_date="2025-09-22",
                region="NA",
                aws_region="us-east-1",
                author="test-author",
                role="arn:aws:iam::123456789012:role/test-role",
                service_name="AtoZ",
                pipeline_version="1.0.0",
                framework_version="1.7-1",
                py_version="py3",
                project_root_folder="project_xgboost_pda",
                source_dir="materials",
                job_type="training",
                label_name="is_abuse",
                processing_entry_point="tabular_preprocessing.py"
            )
            
            # Create step builder
            builder = TabularPreprocessingStepBuilder(config)
            
            # Mock cursus location for monorepo detection
            with patch('pathlib.Path.__file__', str(cursus_dir / "config_base.py")):
                # Test hybrid resolution
                resolved_path = config.get_resolved_script_path()
                self.assertEqual(resolved_path, str(script_file))
                
                # Test step creation with hybrid resolution
                step = builder.create_step()
                self.assertEqual(step.code, str(script_file))
```

### Phase 3: Configuration System Integration (Week 3)

#### **3.1 Update Configuration Classes**

**Files**: All config classes in `src/cursus/steps/configs/`

**Pattern**: Ensure all configuration classes inherit hybrid resolution:

```python
class TabularPreprocessingConfig(ProcessingStepConfigBase):
    """Tabular preprocessing configuration with hybrid path resolution."""
    
    # Existing fields unchanged
    job_type: str = Field(description="Job type for preprocessing")
    label_name: str = Field(description="Label column name")
    
    # Hybrid resolution inherited from ProcessingStepConfigBase
    # - project_root_folder: Tier 1 required field
    # - source_dir: Tier 1 required field
    # - get_resolved_script_path(): Hybrid resolution method
```

#### **3.2 Configuration Validation**

**File**: `test/steps/configs/test_hybrid_config_validation.py`

```python
class TestHybridConfigValidation(unittest.TestCase):
    
    def test_tier1_fields_required(self):
        """Test that Tier 1 fields are required for hybrid resolution."""
        # Test missing project_root_folder
        with self.assertRaises(ValidationError):
            TabularPreprocessingConfig(
                bucket="test-bucket",
                # project_root_folder missing
                source_dir="materials",
                job_type="training",
                label_name="is_abuse"
            )
        
        # Test missing source_dir
        with self.assertRaises(ValidationError):
            TabularPreprocessingConfig(
                bucket="test-bucket",
                project_root_folder="project_xgboost_pda",
                # source_dir missing
                job_type="training",
                label_name="is_abuse"
            )
    
    def test_hybrid_resolution_configuration(self):
        """Test configuration with hybrid resolution fields."""
        config = TabularPreprocessingConfig(
            bucket="test-bucket",
            current_date="2025-09-22",
            region="NA",
            aws_region="us-east-1",
            author="test-author",
            role="arn:aws:iam::123456789012:role/test-role",
            service_name="AtoZ",
            pipeline_version="1.0.0",
            framework_version="1.7-1",
            py_version="py3",
            project_root_folder="project_xgboost_pda",  # Tier 1 required
            source_dir="materials",                     # Tier 1 required
            job_type="training",
            label_name="is_abuse",
            processing_entry_point="tabular_preprocessing.py"
        )
        
        # Verify Tier 1 fields are set
        self.assertEqual(config.project_root_folder, "project_xgboost_pda")
        self.assertEqual(config.source_dir, "materials")
        
        # Verify hybrid resolution methods are available
        self.assertTrue(hasattr(config, 'get_resolved_script_path'))
        self.assertTrue(hasattr(config, 'resolved_processing_source_dir'))
```

### Phase 4: Real-World Validation (Week 4)

#### **4.1 Lambda/MODS Deployment Testing**

**File**: `test/integration/test_lambda_mods_hybrid_resolution.py`

```python
class TestLambdaMODSHybridResolution(unittest.TestCase):
    
    def test_mods_lambda_deployment_scenario(self):
        """Test hybrid resolution in simulated Lambda/MODS environment."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Simulate Lambda filesystem structure
            lambda_var_task = Path(temp_dir) / "var" / "task"
            lambda_var_task.mkdir(parents=True)
            
            lambda_tmp_package = Path(temp_dir) / "tmp" / "buyer_abuse_mods_template"
            lambda_tmp_package.mkdir(parents=True)
            
            # Create cursus package structure
            cursus_dir = lambda_tmp_package / "cursus" / "core" / "base"
            cursus_dir.mkdir(parents=True)
            
            # Create target project structure
            target_dir = lambda_tmp_package / "mods_pipeline_adapter" / "dockers" / "xgboost_atoz"
            target_dir.mkdir(parents=True)
            target_file = target_dir / "tabular_preprocessing.py"
            target_file.write_text("# Mock tabular preprocessing script")
            
            # Create configuration
            config = TabularPreprocessingConfig(
                bucket="test-bucket",
                current_date="2025-09-22",
                region="NA",
                aws_region="us-east-1",
                author="test-author",
                role="arn:aws:iam::123456789012:role/test-role",
                service_name="AtoZ",
                pipeline_version="1.0.0",
                framework_version="1.7-1",
                py_version="py3",
                project_root_folder="mods_pipeline_adapter",
                source_dir="dockers/xgboost_atoz",
                job_type="training",
                label_name="is_abuse",
                processing_entry_point="tabular_preprocessing.py"
            )
            
            # Test hybrid resolution in Lambda context
            with patch('pathlib.Path.__file__', str(cursus_dir / "config_base.py")), \
                 patch('pathlib.Path.cwd', return_value=lambda_var_task):
                
                # Test Package Location Discovery (should succeed)
                resolved_path = config.get_resolved_script_path()
                self.assertEqual(resolved_path, str(target_file))
                
                # Test step builder integration
                builder = TabularPreprocessingStepBuilder(config)
                step = builder.create_step()
                self.assertEqual(step.code, str(target_file))
```

#### **4.2 Development Monorepo Testing**

**File**: `test/integration/test_monorepo_hybrid_resolution.py`

```python
class TestMonorepoHybridResolution(unittest.TestCase):
    
    def test_current_repository_structure(self):
        """Test hybrid resolution with actual repository structure."""
        # Test with actual project structures
        test_cases = [
            {
                "project_root_folder": "project_pytorch_bsm_ext",
                "source_dir": "docker",
                "script_name": "pytorch_training.py"
            },
            {
                "project_root_folder": "project_xgboost_atoz",
                "source_dir": ".",
                "script_name": "xgboost_training.py"
            },
            {
                "project_root_folder": "project_xgboost_pda",
                "source_dir": "materials",
                "script_name": "tabular_preprocessing.py"
            }
        ]
        
        for test_case in test_cases:
            with self.subTest(project=test_case["project_root_folder"]):
                config = TabularPreprocessingConfig(
                    bucket="test-bucket",
                    current_date="2025-09-22",
                    region="NA",
                    aws_region="us-east-1",
                    author="test-author",
                    role="arn:aws:iam::123456789012:role/test-role",
                    service_name="AtoZ",
                    pipeline_version="1.0.0",
                    framework_version="1.7-1",
                    py_version="py3",
                    project_root_folder=test_case["project_root_folder"],
                    source_dir=test_case["source_dir"],
                    job_type="training",
                    label_name="is_abuse",
                    processing_entry_point=test_case["script_name"]
                )
                
                # Test hybrid resolution (should use monorepo detection)
                resolved_path = config.get_resolved_script_path()
                self.assertIsNotNone(resolved_path)
                self.assertTrue(Path(resolved_path).exists())
```

#### **4.3 Pip-Installed Testing**

**File**: `test/integration/test_pip_installed_hybrid_resolution.py`

```python
class TestPipInstalledHybridResolution(unittest.TestCase):
    
    def test_system_wide_installation(self):
        """Test hybrid resolution with system-wide pip installation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Simulate system-wide cursus installation
            system_site_packages = Path(temp_dir) / "usr" / "local" / "lib" / "python3.x" / "site-packages"
            cursus_dir = system_site_packages / "cursus" / "core" / "base"
            cursus_dir.mkdir(parents=True)
            
            # Simulate user project
            user_home = Path(temp_dir) / "home" / "user"
            user_project = user_home / "my_ml_project"
            scripts_dir = user_project / "dockers" / "xgboost_atoz"
            scripts_dir.mkdir(parents=True)
            script_file = scripts_dir / "preprocessing.py"
            script_file.write_text("# Mock preprocessing script")
            
            # Create configuration
            config = TabularPreprocessingConfig(
                bucket="test-bucket",
                current_date="2025-09-22",
                region="NA",
                aws_region="us-east-1",
                author="test-author",
                role="arn:aws:iam::123456789012:role/test-role",
                service_name="AtoZ",
                pipeline_version="1.0.0",
                framework_version="1.7-1",
                py_version="py3",
                project_root_folder="my_ml_project",
                source_dir="dockers/xgboost_atoz",
                job_type="training",
                label_name="is_abuse",
                processing_entry_point="preprocessing.py"
            )
            
            # Test hybrid resolution in pip-installed context
            with patch('pathlib.Path.__file__', str(cursus_dir / "config_base.py")), \
                 patch('pathlib.Path.cwd', return_value=user_project):
                
                # Test Working Directory Discovery (should succeed as fallback)
                resolved_path = config.get_resolved_script_path()
                self.assertEqual(resolved_path, str(script_file))
                
                # Test step builder integration
                builder = TabularPreprocessingStepBuilder(config)
                step = builder.create_step()
                self.assertEqual(step.code, str(script_file))
```

## Error Handling and Logging Implementation

### **Enhanced Error Handling**

**File**: `src/cursus/core/base/config_base.py` (additions)

```python
def _resolve_source_dir_with_fallbacks(self, project_root_folder: str, relative_path: str) -> Optional[str]:
    """Resolve path with comprehensive hybrid fallback strategy."""
    if not relative_path:
        return None
    
    try:
        # Primary: Hybrid resolution algorithm
        resolved = self._resolve_source_dir(project_root_folder, relative_path)
        if resolved:
            logger.info(f"Hybrid resolution succeeded: {resolved}")
            return resolved
    except Exception as e:
        logger.warning(f"Hybrid resolution failed: {e}")
    
    try:
        # Fallback 1: Direct working directory relative (without project_root_folder)
        cwd_path = Path.cwd() / relative_path
        if cwd_path.exists():
            logger.info(f"Using direct working directory fallback: {cwd_path}")
            return str(cwd_path)
    except Exception as e:
        logger.warning(f"Direct working directory fallback failed: {e}")
    
    try:
        # Fallback 2: Working directory with project_root_folder
        if project_root_folder:
            project_path = Path.cwd() / project_root_folder / relative_path
            if project_path.exists():
                logger.info(f"Using project-specific working directory fallback: {project_path}")
                return str(project_path)
    except Exception as e:
        logger.warning(f"Project-specific working directory fallback failed: {e}")
    
    # Fallback 3: Return relative path as-is for external resolution
    logger.warning(f"All hybrid resolution strategies failed, returning relative path: {relative_path}")
    return relative_path
```

### **Comprehensive Logging**

**File**: `src/cursus/core/base/config_base.py` (additions)

```python
def _resolve_source_dir_with_detailed_logging(self, project_root_folder: str, relative_path: str) -> Optional[str]:
    """Hybrid path resolution with detailed execution logging."""
    logger.info(f"Starting hybrid path resolution: project_root_folder='{project_root_folder}', relative_path='{relative_path}'")
    
    if not relative_path:
        logger.warning("Empty relative_path provided, returning None")
        return None
    
    # Strategy 1: Package Location Discovery
    logger.info("Attempting Strategy 1: Package Location Discovery")
    resolved = self._package_location_discovery(project_root_folder, relative_path)
    if resolved:
        logger.info(f"Hybrid resolution completed successfully via Package Location Discovery: {resolved}")
        return resolved
    
    logger.info("Strategy 1 failed, attempting Strategy 2: Working Directory Discovery")
    
    # Strategy 2: Working Directory Discovery
    resolved = self._working_directory_discovery(project_root_folder, relative_path)
    if resolved:
        logger.info(f"Hybrid resolution completed successfully via Working Directory Discovery: {resolved}")
        return resolved
    
    logger.warning(f"Hybrid resolution failed - both strategies unsuccessful for project_root_folder='{project_root_folder}', relative_path='{relative_path}'")
    return None
```

## Performance Monitoring and Metrics

### **Resolution Performance Tracking**

**File**: `src/cursus/core/base/config_base.py` (additions)

```python
import time
from typing import Dict, Any

class HybridResolutionMetrics:
    """Track hybrid resolution performance metrics."""
    
    def __init__(self):
        self.strategy_1_success_count = 0
        self.strategy_2_success_count = 0
        self.total_resolution_attempts = 0
        self.resolution_times = []
        self.failure_count = 0
    
    def record_strategy_1_success(self, resolution_time: float):
        """Record successful Package Location Discovery."""
        self.strategy_1_success_count += 1
        self.total_resolution_attempts += 1
        self.resolution_times.append(resolution_time)
    
    def record_strategy_2_success(self, resolution_time: float):
        """Record successful Working Directory Discovery."""
        self.strategy_2_success_count += 1
        self.total_resolution_attempts += 1
        self.resolution_times.append(resolution_time)
    
    def record_failure(self, resolution_time: float):
        """Record resolution failure."""
        self.failure_count += 1
        self.total_resolution_attempts += 1
        self.resolution_times.append(resolution_time)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if self.total_resolution_attempts == 0:
            return {"status": "no_data"}
        
        return {
            "strategy_1_success_rate": self.strategy_1_success_count / self.total_resolution_attempts,
            "strategy_2_fallback_rate": self.strategy_2_success_count / self.total_resolution_attempts,
            "failure_rate": self.failure_count / self.total_resolution_attempts,
            "average_resolution_time": sum(self.resolution_times) / len(self.resolution_times),
            "total_attempts": self.total_resolution_attempts
        }

# Global metrics instance
_hybrid_resolution_metrics = HybridResolutionMetrics()

def _resolve_source_dir_with_metrics(self, project_root_folder: str, relative_path: str) -> Optional[str]:
    """Hybrid path resolution with performance metrics tracking."""
    start_time = time.time()
    
    try:
        # Strategy 1: Package Location Discovery
        resolved = self._package_location_discovery(project_root_folder, relative_path)
        if resolved:
            resolution_time = time.time() - start_time
            _hybrid_resolution_metrics.record_strategy_1_success(resolution_time)
            return resolved
        
        # Strategy 2: Working Directory Discovery
        resolved = self._working_directory_discovery(project_root_folder, relative_path)
        if resolved:
            resolution_time = time.time() - start_time
            _hybrid_resolution_metrics.record_strategy_2_success(resolution_time)
            return resolved
        
        # Resolution failed
        resolution_time = time.time() - start_time
        _hybrid_resolution_metrics.record_failure(resolution_time)
        return None
        
    except Exception as e:
        resolution_time = time.time() - start_time
        _hybrid_resolution_metrics.record_failure(resolution_time)
        logger.error(f"Hybrid resolution error: {e}")
        return None

def get_hybrid_resolution_metrics() -> Dict[str, Any]:
    """Get current hybrid resolution performance metrics."""
    return _hybrid_resolution_metrics.get_metrics()
```

## Migration and Deployment Strategy

### **Backward Compatibility**

**File**: `src/cursus/core/base/config_base.py` (additions)

```python
def get_script_path_with_hybrid_fallback(self) -> Optional[str]:
    """Get script path with hybrid resolution and backward compatibility."""
    # Try hybrid resolution first
    hybrid_path = self.get_resolved_script_path()
    if hybrid_path:
        return hybrid_path
    
    # Fallback to existing get_script_path for backward compatibility
    try:
        legacy_path = self.get_script_path()
        if legacy_path and Path(legacy_path).exists():
            logger.info(f"Using legacy path resolution: {legacy_path}")
            return legacy_path
    except Exception as e:
        logger.warning(f"Legacy path resolution failed: {e}")
    
    return None
```

### **Gradual Rollout Configuration**

**File**: `src/cursus/core/base/config_base.py` (additions)

```python
import os

class HybridResolutionConfig:
    """Configuration for hybrid resolution rollout."""
    
    @staticmethod
    def is_hybrid_resolution_enabled() -> bool:
        """Check if hybrid resolution is enabled via environment variable."""
        return os.getenv("CURSUS_HYBRID_RESOLUTION_ENABLED", "true").lower() == "true"
    
    @staticmethod
    def get_hybrid_resolution_mode() -> str:
        """Get hybrid resolution mode: 'full', 'fallback_only', 'disabled'."""
        return os.getenv("CURSUS_HYBRID_RESOLUTION_MODE", "full")

def _resolve_source_dir_with_rollout_control(self, project_root_folder: str, relative_path: str) -> Optional[str]:
    """Hybrid path resolution with rollout control."""
    if not HybridResolutionConfig.is_hybrid_resolution_enabled():
        # Hybrid resolution disabled, use legacy behavior
        return None
    
    mode = HybridResolutionConfig.get_hybrid_resolution_mode()
    
    if mode == "full":
        # Full hybrid resolution
        return self._resolve_source_dir(project_root_folder, relative_path)
    elif mode == "fallback_only":
        # Only use Working Directory Discovery
        return self._working_directory_discovery(project_root_folder, relative_path)
    else:
        # Disabled
        return None
```

## Success Metrics and Monitoring

### **Implementation Success Criteria**

#### **Phase 1 Success Metrics**
- [ ] All core hybrid algorithm tests pass (100% test coverage)
- [ ] Package Location Discovery works in bundled scenarios
- [ ] Working Directory Discovery works as fallback
- [ ] Monorepo structure detection functions correctly
- [ ] Performance metrics show <10ms average resolution time

#### **Phase 2 Success Metrics**
- [ ] All step builders updated to use hybrid resolution
- [ ] Step builder integration tests pass (100% coverage)
- [ ] Backward compatibility maintained (no breaking changes)
- [ ] Hybrid resolution success rate >95% in development environment

#### **Phase 3 Success Metrics**
- [ ] All configuration classes support Tier 1 fields
- [ ] Configuration validation enforces required fields
- [ ] Hybrid resolution methods available on all configs
- [ ] Configuration serialization/deserialization works with new fields

#### **Phase 4 Success Metrics**
- [ ] Lambda/MODS deployment tests pass
- [ ] Development monorepo tests pass with actual project structures
- [ ] Pip-installed deployment tests pass
- [ ] End-to-end integration tests pass across all scenarios
- [ ] Performance metrics show Strategy 1 success rate >80%

### **Production Monitoring**

#### **Key Performance Indicators (KPIs)**
- **Strategy 1 Success Rate**: Target >80% (Package Location Discovery)
- **Strategy 2 Fallback Rate**: Target <20% (Working Directory Discovery)
- **Resolution Failure Rate**: Target <1%
- **Average Resolution Time**: Target <10ms
- **Configuration Portability**: Same config works across all deployment scenarios

#### **Monitoring Dashboard Metrics**
```python
# Example monitoring integration
def log_hybrid_resolution_metrics():
    """Log hybrid resolution metrics for monitoring dashboard."""
    metrics = get_hybrid_resolution_metrics()
    
    # Log to monitoring system (e.g., CloudWatch, DataDog)
    logger.info("HYBRID_RESOLUTION_METRICS", extra={
        "strategy_1_success_rate": metrics.get("strategy_1_success_rate", 0),
        "strategy_2_fallback_rate": metrics.get("strategy_2_fallback_rate", 0),
        "failure_rate": metrics.get("failure_rate", 0),
        "average_resolution_time_ms": metrics.get("average_resolution_time", 0) * 1000,
        "total_attempts": metrics.get("total_attempts", 0)
    })
```

## Conclusion

This implementation plan provides a comprehensive roadmap for implementing the Hybrid Strategy Deployment Path Resolution system in cursus. The phased approach ensures:

### **Technical Excellence**
- **Robust hybrid algorithm** - Package Location First with Working Directory Discovery fallback
- **Universal configuration portability** - same config works across all deployment scenarios
- **Comprehensive error handling** - multiple fallback strategies and detailed logging
- **Performance monitoring** - metrics tracking for optimization and troubleshooting

### **Implementation Safety**
- **Backward compatibility** - existing configurations continue to work
- **Gradual rollout** - controlled deployment with feature flags
- **Comprehensive testing** - unit, integration, and end-to-end test coverage
- **Real-world validation** - testing with actual deployment scenarios

### **User Experience**
- **Zero configuration changes** - users benefit automatically from hybrid resolution
- **Transparent operation** - hybrid resolution works behind the scenes
- **Improved reliability** - eliminates deployment-specific path failures
- **Enhanced portability** - configurations work universally across environments

The implementation transforms cursus from a deployment-fragile system into a truly universal framework that works seamlessly across development, staging, and production environments through the hybrid path resolution algorithm.

## References

### **Design Documents**
- **[Hybrid Strategy Deployment Path Resolution Design](../1_design/hybrid_strategy_deployment_path_resolution_design.md)** - Complete technical design and architecture

### **Related Implementation Plans**
- **[2025-09-22 Simplified Path File-Based Resolution Implementation Plan](./2025-09-22_simplified_path_file_based_resolution_implementation_plan.md)** - Previous implementation approach

### **Analysis Documents**
- **[MODS Pipeline Path Resolution Error Analysis](../.internal/mods_pipeline_path_resolution_error_analysis.md)** - Error analysis that motivated this solution

### **Configuration System Documentation**
- **[Three-Tier Config Design](../1_design/config_tiered_design.md)** - Configuration architecture patterns
- **[Config Field Categorization Consolidated](../1_design/config_field_categorization_consolidated.md)** - Field management principles

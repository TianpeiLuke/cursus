---
tags:
  - design
  - architecture
  - portability
  - configuration
  - path_resolution
  - workspace_aware
keywords:
  - config portability
  - path resolution
  - workspace aware paths
  - relative path system
  - step catalog integration
  - deployment agnostic
  - package portability
topics:
  - configuration path portability
  - workspace-aware path resolution
  - step catalog path discovery
  - relative path configuration system
  - deployment-agnostic configuration
language: python
date of note: 2025-09-20
---

# Configuration Portability Path Resolution Design

## Executive Summary

This document presents a comprehensive design for making cursus configuration files portable across different deployment environments by replacing absolute paths with a workspace-aware, relative path resolution system. The design addresses the critical portability issue where saved configuration JSON files contain hardcoded absolute paths that become invalid when used in different systems, different package installations, or different deployment environments.

### Problem Statement

The current configuration system suffers from a fundamental portability flaw:

**Current Issue**: Configuration JSON files contain absolute paths like:
```json
{
  "source_dir": "/home/ec2-user/SageMaker/Cursus/dockers/xgboost_atoz",
  "processing_source_dir": "/home/ec2-user/SageMaker/Cursus/dockers/xgboost_atoz/scripts",
  "script_path": "/home/ec2-user/SageMaker/Cursus/dockers/xgboost_atoz/scripts/tabular_preprocessing.py"
}
```

**Portability Problems**:
1. **Cross-System Incompatibility**: Paths valid on one system are invalid on another
2. **Package Installation Issues**: PyPI-installed packages have different root directories
3. **Container Deployment Failures**: Docker containers use different filesystem layouts
4. **Workspace Isolation Breaks**: Different developers/projects can't share configurations
5. **CI/CD Pipeline Failures**: Automated systems can't use saved configurations

### Solution Overview

The design introduces a **Workspace-Aware Relative Path Resolution System** that:

1. **Stores Relative Paths**: Configuration JSON files contain only relative paths and workspace identifiers
2. **Runtime Path Resolution**: Absolute paths are resolved at runtime using workspace context
3. **Step Catalog Integration**: Leverages the unified step catalog for intelligent path discovery
4. **Deployment Agnostic**: Works consistently across all deployment environments
5. **Backward Compatible**: Existing absolute path configurations continue to work

## Current System Analysis

### Path Usage in Configuration System

The current system uses absolute paths in multiple layers:

#### 1. Base Configuration Layer (`BasePipelineConfig`)
```python
# Current implementation in config_base.py
def get_script_path(self, default_path: Optional[str] = None) -> Optional[str]:
    """Get script path, preferring contract-defined path if available."""
    contract = self.get_script_contract()
    if contract and hasattr(contract, "script_path"):
        return contract.script_path  # Often absolute path
    
    if hasattr(self, "script_path"):
        return self.script_path  # Often absolute path
    
    return default_path
```

#### 2. Processing Step Configuration Layer (`ProcessingStepConfigBase`)
```python
# Current implementation in config_processing_step_base.py
@property
def script_path(self) -> Optional[str]:
    """Get the full path to the processing script if entry point is provided."""
    if self.processing_entry_point is None:
        return None

    if self._script_path is None:
        effective_source = self.effective_source_dir
        if effective_source is None:
            return None

        if effective_source.startswith("s3://"):
            self._script_path = f"{effective_source.rstrip('/')}/{self.processing_entry_point}"
        else:
            # PROBLEM: Creates absolute path
            self._script_path = str(Path(effective_source) / self.processing_entry_point)

    return self._script_path
```

#### 3. Step Builder Usage (`XGBoostTrainingStepBuilder`)
```python
# Current implementation in builder_xgboost_training_step.py
def _create_estimator(self, output_path=None) -> XGBoost:
    return XGBoost(
        entry_point=self.config.training_entry_point,
        source_dir=self.config.source_dir,  # Absolute path used directly
        # ... other parameters
    )
```

### Portability Issues in Practice

#### Issue 1: Saved Configuration Non-Portability
```json
// Saved on System A
{
  "source_dir": "/home/ec2-user/SageMaker/Cursus/dockers/xgboost_atoz",
  "processing_source_dir": "/home/ec2-user/SageMaker/Cursus/dockers/xgboost_atoz/scripts"
}

// Fails on System B (different user)
{
  "source_dir": "/home/alice/projects/cursus/dockers/xgboost_atoz",  // Different path
  "processing_source_dir": "/home/alice/projects/cursus/dockers/xgboost_atoz/scripts"
}

// Fails on PyPI Installation
{
  "source_dir": "/usr/local/lib/python3.9/site-packages/cursus/dockers/xgboost_atoz",
  "processing_source_dir": "/usr/local/lib/python3.9/site-packages/cursus/dockers/xgboost_atoz/scripts"
}
```

#### Issue 2: Step Creation Failures
```python
# Current step creation process
def create_step(self, **kwargs):
    # This fails when config contains invalid absolute paths
    estimator = XGBoost(
        source_dir=self.config.source_dir,  # "/home/ec2-user/..." - doesn't exist on new system
        entry_point=self.config.training_entry_point,
    )
```

## Proposed Solution Architecture

### Core Design Principle: Direct Enhancement of Existing Config Classes

Following the **code redundancy evaluation guide** and **config tiered design**, the solution uses the existing three-tier config architecture without creating new classes. The key insight is to enhance the existing `BasePipelineConfig` and `ProcessingStepConfigBase` directly:

#### **Single Design Principle: Automatic Path Conversion in Existing Config Classes**

Instead of creating new classes or complex systems, we simply:

1. **Keep existing `source_dir` field unchanged** (Tier 1 or Tier 2 field) - no breaking changes
2. **Add automatic path conversion** using private fields (Tier 3) 
3. **Provide read-only property** for step builders to access the portable path

This follows the established pattern where configs handle their own field derivations using private fields and properties, with zero impact on existing functionality.

#### **Implementation Pattern: Direct Enhancement**

```python
class BasePipelineConfig(BaseModel):
    """Base configuration with automatic path conversion for portability."""
    
    # Existing field - UNCHANGED (no breaking changes)
    source_dir: Optional[str] = Field(
        default=None,
        description="Common source directory for scripts if applicable. Can be overridden by step configs."
    )
    
    # NEW: Private derived field (Tier 3) - stores portable relative path
    _portable_source_dir: Optional[str] = PrivateAttr(default=None)
    
    # NEW: Property for step builders to use portable paths
    @property
    def portable_source_dir(self) -> Optional[str]:
        """Get source directory as relative path for portability."""
        if self.source_dir is None:
            return None
            
        if self._portable_source_dir is None:
            self._portable_source_dir = self._convert_to_relative_path(self.source_dir)
        
        return self._portable_source_dir
    
    # NEW: Path conversion method
    def _convert_to_relative_path(self, path: str) -> str:
        """
        Convert absolute path to relative path based on config/builder relationship.
        
        This method converts absolute filesystem paths (like source_dir, processing_source_dir)
        from their absolute form to relative paths that work across different deployment environments.
        
        Args:
            path (str): The absolute filesystem path to convert. This is typically:
                       - source_dir: "/home/user/cursus/dockers/xgboost_atoz"
                       - processing_source_dir: "/home/user/cursus/dockers/xgboost_atoz/scripts"
                       - Any other absolute path pointing to cursus resources
        
        Returns:
            str: Relative path from the step builder's perspective. Examples:
                 - "../../dockers/xgboost_atoz" (for source_dir)
                 - "../../dockers/xgboost_atoz/scripts" (for processing_source_dir)
        
        Path Resolution Strategy:
            1. Determines the config file location using inspect.getfile()
            2. Calculates the builders directory (sibling to configs directory)
            3. Converts the absolute path to be relative from the builders directory
            4. This ensures step builders can resolve paths correctly at runtime
        
        Example Conversion:
            Input:  "/home/user/cursus/dockers/xgboost_atoz"
            Config: "/home/user/cursus/src/cursus/steps/configs/config_xgboost.py"
            Builder: "/home/user/cursus/src/cursus/steps/builders/builder_xgboost.py"
            Output: "../../dockers/xgboost_atoz" (relative to builder location)
        """
        if not path or not Path(path).is_absolute():
            return path  # Already relative, keep as-is
        
        try:
            # Directory structure analysis:
            # Config location: src/cursus/steps/configs/config_*.py
            # Builder location: src/cursus/steps/builders/builder_*.py
            # Target: Make path relative to builders directory for step builder usage
            
            config_file = Path(inspect.getfile(self.__class__))
            config_dir = config_file.parent      # .../steps/configs/
            steps_dir = config_dir.parent        # .../steps/
            builders_dir = steps_dir / "builders" # .../steps/builders/
            
            # Convert the absolute resource path to be relative from builders directory
            # This allows step builders to resolve the path correctly at runtime
            abs_path = Path(path)
            relative_path = abs_path.relative_to(builders_dir)
            
            return str(relative_path)
            
        except (ValueError, OSError):
            # If direct conversion fails (e.g., paths don't share common structure),
            # try the common parent approach as fallback
            return self._convert_via_common_parent(path)
    
    # NEW: Fallback conversion method
    def _convert_via_common_parent(self, path: str) -> str:
        """
        Fallback: convert using common parent directory when direct builder-relative conversion fails.
        
        This method is used when the primary path conversion strategy fails, typically when:
        - The target path and builders directory don't share a common structure
        - The path is outside the expected cursus directory structure
        - There are permission issues or path resolution errors
        
        Args:
            path (str): The absolute filesystem path to convert. Examples:
                       - "/different/root/cursus/dockers/xgboost_atoz" (different installation)
                       - "/usr/local/lib/python3.9/site-packages/cursus/dockers/xgboost_atoz" (PyPI install)
                       - "/app/cursus/dockers/xgboost_atoz" (container deployment)
        
        Returns:
            str: Relative path using common parent approach, or original path if conversion fails.
                 Examples:
                 - "../../../dockers/xgboost_atoz" (when common parent is cursus root)
                 - "../../../../dockers/xgboost_atoz" (when deeper nesting)
        
        Algorithm:
            1. Find the deepest common parent directory between config location and target path
            2. Calculate how many levels to go up from config directory to common parent
            3. Calculate path from common parent down to target
            4. Combine with appropriate number of "../" prefixes
        
        Example Conversion:
            Config:  "/home/user/cursus/src/cursus/steps/configs/config_xgboost.py"
            Target:  "/home/user/cursus/dockers/xgboost_atoz"
            Common:  "/home/user/cursus" (cursus root)
            
            Steps:
            - Config to common: 4 levels up (configs → steps → cursus → src → cursus root)
            - Common to target: "dockers/xgboost_atoz"
            - Result: "../../../../dockers/xgboost_atoz"
        
        Edge Cases Handled:
            - No common parent found: Returns original path
            - Permission errors: Returns original path
            - Invalid path structures: Returns original path
        """
        try:
            config_file = Path(inspect.getfile(self.__class__))
            config_dir = config_file.parent
            abs_path = Path(path)
            
            # Find the deepest common parent directory between config and target paths
            # This handles cases where paths have different structures but share a root
            common_parent = self._find_common_parent(abs_path, config_dir)
            if common_parent:
                # Calculate the path from config directory up to the common parent
                # This tells us how many "../" we need
                config_to_common = config_dir.relative_to(common_parent)
                
                # Calculate the path from common parent down to the target
                # This is the path after all the "../" navigation
                common_to_target = abs_path.relative_to(common_parent)
                
                # Build the complete relative path:
                # - One "../" for each directory level from config to common parent
                # - Followed by the path from common parent to target
                up_levels = len(config_to_common.parts)
                relative_parts = ['..'] * up_levels + list(common_to_target.parts)
                
                return str(Path(*relative_parts))
        
        except (ValueError, OSError) as e:
            # Handle cases where:
            # - Paths don't have a common parent (ValueError)
            # - File system access issues (OSError)
            # - Path resolution failures
            pass
        
        # Final fallback: return original absolute path
        # This ensures the system continues to work even if path conversion fails
        # Step builders will receive the absolute path and may still function
        return path
    
    # NEW: Helper method to find common parent
    def _find_common_parent(self, path1: Path, path2: Path) -> Optional[Path]:
        """Find common parent directory of two paths."""
        try:
            parts1 = path1.parts
            parts2 = path2.parts
            
            common_parts = []
            for p1, p2 in zip(parts1, parts2):
                if p1 == p2:
                    common_parts.append(p1)
                else:
                    break
            
            if common_parts:
                return Path(*common_parts)
        except Exception:
            pass
        
        return None

    # ENHANCED: Include portable paths in serialization
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include both original and portable paths."""
        data = super().model_dump(**kwargs)
        
        # Add portable path as additional field - keep original source_dir intact
        if self.portable_source_dir is not None:
            data["portable_source_dir"] = self.portable_source_dir
        
        return data
```

```python
class ProcessingStepConfigBase(BasePipelineConfig):
    """Processing configuration with automatic path conversion for portability."""
    
    # Existing field - UNCHANGED (no breaking changes)
    processing_source_dir: Optional[str] = Field(
        default=None,
        description="Source directory for processing scripts. Falls back to base source_dir if not provided."
    )
    
    # NEW: Private derived field (Tier 3) - stores portable relative path
    _portable_processing_source_dir: Optional[str] = PrivateAttr(default=None)
    
    # NEW: Property for step builders to use portable paths
    @property
    def portable_processing_source_dir(self) -> Optional[str]:
        """Get processing source directory as relative path for portability."""
        if self.processing_source_dir is None:
            return None
            
        if self._portable_processing_source_dir is None:
            self._portable_processing_source_dir = self._convert_to_relative_path(self.processing_source_dir)
        
        return self._portable_processing_source_dir
    
    # NEW: Portable version of effective source directory
    @property
    def portable_effective_source_dir(self) -> Optional[str]:
        """Get effective source directory as relative path for step builders to use."""
        return self.portable_processing_source_dir or self.portable_source_dir
    
    # ENHANCED: Include portable paths in serialization
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to include both original and portable paths."""
        data = super().model_dump(**kwargs)
        
        # Add portable path as additional field - keep original processing_source_dir intact
        if self.portable_processing_source_dir is not None:
            data["portable_processing_source_dir"] = self.portable_processing_source_dir
        
        return data
```

### System Architecture: Zero-Impact Enhancement

Following the **15-25% redundancy target** from the code redundancy guide, we avoid over-engineering and enhance existing classes directly:

#### **Key Benefits of Direct Enhancement**

1. **Zero Breaking Changes**: All existing code continues to work exactly as before
2. **Transparent Enhancement**: Users don't need to change anything - portability is automatic
3. **Backward Compatibility**: Existing configurations continue to work without modification
4. **Minimal Code Changes**: Only adds new functionality, doesn't modify existing behavior
5. **Simple Migration**: No migration needed - enhancement is additive only

#### **Step Builder Changes: Minimal Single-Line Updates**

Based on analysis of real step builder implementations in `src/cursus/steps/builders/`, the changes required are minimal and follow a consistent pattern across all step builder types.

**Current Path Usage Patterns in Step Builders:**

Step builders currently use absolute paths directly from configuration in several key locations:

1. **Training Step Builders** (`XGBoostTrainingStepBuilder`, `PyTorchTrainingStepBuilder`):
   ```python
   # Current implementation in _create_estimator()
   return XGBoost(
       entry_point=self.config.training_entry_point,
       source_dir=self.config.source_dir,  # ← Uses absolute path directly
       framework_version=self.config.framework_version,
       # ... other parameters
   )
   ```

2. **Processing Step Builders** (`TabularPreprocessingStepBuilder`):
   ```python
   # Current implementation in create_step()
   script_path = self.config.get_script_path()  # ← Returns absolute path
   
   step = ProcessingStep(
       name=step_name,
       processor=processor,
       code=script_path,  # ← Uses absolute path directly
       # ... other parameters
   )
   ```

**Required Changes: Single Line Updates**

The portable path resolution requires only **one line changes** in each step builder:

1. **Training Step Builders Enhancement**:
   ```python
   # BEFORE: Uses absolute path
   def _create_estimator(self, output_path=None) -> XGBoost:
       return XGBoost(
           entry_point=self.config.training_entry_point,
           source_dir=self.config.source_dir,  # Absolute path
           framework_version=self.config.framework_version,
           # ... other parameters
       )
   
   # AFTER: Uses portable path with fallback
   def _create_estimator(self, output_path=None) -> XGBoost:
       return XGBoost(
           entry_point=self.config.training_entry_point,
           source_dir=self.config.portable_source_dir or self.config.source_dir,  # ← Single line change
           framework_version=self.config.framework_version,
           # ... other parameters
       )
   ```

2. **Processing Step Builders Enhancement**:
   ```python
   # BEFORE: Uses absolute script path
   def create_step(self, **kwargs) -> ProcessingStep:
       # ... setup code ...
       script_path = self.config.get_script_path()  # Absolute path
       
       step = ProcessingStep(
           name=step_name,
           processor=processor,
           code=script_path,  # Absolute path
           # ... other parameters
       )
   
   # AFTER: Uses portable script path with fallback
   def create_step(self, **kwargs) -> ProcessingStep:
       # ... setup code ...
       script_path = self.config.get_portable_script_path() or self.config.get_script_path()  # ← Single line change
       
       step = ProcessingStep(
           name=step_name,
           processor=processor,
           code=script_path,  # Portable path
           # ... other parameters
       )
   ```

**Step Builder Migration Pattern:**

All step builders follow the same simple migration pattern:

```python
# Pattern: Replace direct config path usage with portable path + fallback
# BEFORE:
some_path = self.config.path_field

# AFTER:
some_path = self.config.portable_path_field or self.config.path_field
```

**Affected Step Builders and Required Changes:**

Based on comprehensive analysis of all step builders in `src/cursus/steps/builders/`, here are the specific changes needed for each:

### **Training Step Builders (Source Directory Changes)**

1. **`XGBoostTrainingStepBuilder`** - **REQUIRES UPDATE**
   - **Location**: `_create_estimator()` method
   - **Current**: `source_dir=self.config.source_dir`
   - **Change**: `source_dir=self.config.portable_source_dir or self.config.source_dir`
   - **Impact**: Enables portable paths for XGBoost training containers

2. **`PyTorchTrainingStepBuilder`** - **REQUIRES UPDATE**
   - **Location**: `_create_estimator()` method
   - **Current**: `source_dir=self.config.source_dir`
   - **Change**: `source_dir=self.config.portable_source_dir or self.config.source_dir`
   - **Impact**: Enables portable paths for PyTorch training containers

3. **`DummyTrainingStepBuilder`** - **REQUIRES UPDATE** (if exists)
   - **Location**: Similar pattern in estimator creation
   - **Change**: Same pattern as above training steps

### **Model Step Builders (Source Directory Changes)**

4. **`XGBoostModelStepBuilder`** - **REQUIRES UPDATE**
   - **Location**: `_create_model()` method
   - **Current**: `source_dir=self.config.source_dir`
   - **Change**: `source_dir=self.config.portable_source_dir or self.config.source_dir`
   - **Impact**: Enables portable paths for XGBoost model inference containers

5. **`PyTorchModelStepBuilder`** - **REQUIRES UPDATE**
   - **Location**: `_create_model()` method
   - **Current**: `source_dir=self.config.source_dir`
   - **Change**: `source_dir=self.config.portable_source_dir or self.config.source_dir`
   - **Impact**: Enables portable paths for PyTorch model inference containers

### **Processing Step Builders (Script Path Changes)**

6. **`TabularPreprocessingStepBuilder`** - **REQUIRES UPDATE**
   - **Location**: `create_step()` method
   - **Current**: `script_path = self.config.get_script_path()`
   - **Change**: `script_path = self.config.get_portable_script_path() or self.config.get_script_path()`
   - **Impact**: Enables portable script paths for preprocessing

7. **`ModelCalibrationStepBuilder`** - **REQUIRES UPDATE**
   - **Location**: `create_step()` method
   - **Current**: `script_path = self.config.get_script_path()`
   - **Change**: `script_path = self.config.get_portable_script_path() or self.config.get_script_path()`
   - **Impact**: Enables portable script paths for model calibration

8. **`PackageStepBuilder`** - **REQUIRES UPDATE**
   - **Location**: `create_step()` method AND `_get_inputs()` method
   - **Current**: 
     - `script_path = self.config.get_script_path()`
     - `inference_scripts_path = self.config.source_dir`
   - **Change**: 
     - `script_path = self.config.get_portable_script_path() or self.config.get_script_path()`
     - `inference_scripts_path = self.config.portable_source_dir or self.config.source_dir`
   - **Impact**: Enables portable paths for both packaging script and inference scripts

9. **`CradleDataLoadingStepBuilder`** - **LIKELY REQUIRES UPDATE**
   - **Expected Location**: Script path usage in `create_step()`
   - **Expected Change**: Same script path pattern as other processing steps
   - **Impact**: Enables portable script paths for data loading

10. **`CurrencyConversionStepBuilder`** - **LIKELY REQUIRES UPDATE**
    - **Expected Location**: Script path usage in `create_step()`
    - **Expected Change**: Same script path pattern as other processing steps
    - **Impact**: Enables portable script paths for currency conversion

11. **`RiskTableMappingStepBuilder`** - **LIKELY REQUIRES UPDATE**
    - **Expected Location**: Script path usage in `create_step()`
    - **Expected Change**: Same script path pattern as other processing steps
    - **Impact**: Enables portable script paths for risk table mapping

12. **`PayloadStepBuilder`** - **LIKELY REQUIRES UPDATE**
    - **Expected Location**: Script path usage in `create_step()`
    - **Expected Change**: Same script path pattern as other processing steps
    - **Impact**: Enables portable script paths for payload processing

### **Transform Step Builders (No Changes Required)**

13. **`BatchTransformStepBuilder`** - **NO CHANGES REQUIRED**
    - **Reason**: Uses model references from dependencies, not direct file paths
    - **Current Behavior**: Already portable through dependency resolution
    - **Impact**: No action needed

### **Evaluation Step Builders (Script Path Changes)**

14. **`XGBoostModelEvalStepBuilder`** - **LIKELY REQUIRES UPDATE**
    - **Expected Location**: Script path usage in `create_step()`
    - **Expected Change**: Same script path pattern as other processing steps
    - **Impact**: Enables portable script paths for model evaluation

15. **`RegistrationStepBuilder`** - **LIKELY REQUIRES UPDATE**
    - **Expected Location**: Script path usage in `create_step()`
    - **Expected Change**: Same script path pattern as other processing steps
    - **Impact**: Enables portable script paths for model registration

### **Summary of Changes Required**

**Confirmed Updates Needed (8 step builders):**
- 2 Training step builders: XGBoost, PyTorch
- 2 Model step builders: XGBoost, PyTorch  
- 4 Processing step builders: Tabular preprocessing, Model calibration, Package, and others

**Pattern-Based Updates Likely Needed (7 step builders):**
- 5 Processing step builders: Cradle data loading, Currency conversion, Risk table mapping, Payload, XGBoost model eval
- 1 Registration step builder
- 1 Dummy training step builder (if exists)

**No Changes Required (1 step builder):**
- Batch transform step builder (uses dependency resolution)

**Total Impact**: 15 out of 16 step builders require updates, with consistent patterns making implementation straightforward.

**Benefits of This Approach:**

1. **Minimal Code Changes**: Only 1-2 lines per step builder
2. **Automatic Fallback**: If portable path conversion fails, uses original absolute path
3. **Zero Breaking Changes**: Existing functionality preserved completely
4. **Universal Compatibility**: Works across all deployment environments
5. **Transparent Operation**: Users see no difference in behavior

**Example: Complete XGBoost Training Step Builder Update**

```python
class XGBoostTrainingStepBuilder(StepBuilderBase):
    # ... existing code unchanged ...
    
    def _create_estimator(self, output_path=None) -> XGBoost:
        """Creates and configures the XGBoost estimator."""
        return XGBoost(
            entry_point=self.config.training_entry_point,
            source_dir=self.config.portable_source_dir or self.config.source_dir,  # ← Only change needed
            framework_version=self.config.framework_version,
            py_version=self.config.py_version,
            role=self.role,
            instance_type=self.config.training_instance_type,
            instance_count=self.config.training_instance_count,
            volume_size=self.config.training_volume_size,
            base_job_name=self._generate_job_name(),
            sagemaker_session=self.session,
            output_path=output_path,
            environment=self._get_environment_variables(),
        )
    
    # ... rest of the class unchanged ...
```

This single line change enables the step builder to automatically use portable relative paths when available, while maintaining complete backward compatibility with existing absolute path configurations.

### Migration Strategy: Direct Enhancement with Zero Breaking Changes

#### **Phase 1: Enhance Existing Config Classes (Zero Impact)**

Following the **code redundancy evaluation guide**, we enhance existing classes directly without creating new systems:

1. **Enhance `BasePipelineConfig` Directly**
   - Add `_portable_source_dir` private field (Tier 3)
   - Add `portable_source_dir` property for step builders
   - Add path conversion methods
   - Enhance `model_dump()` to include portable paths
   - **No changes to existing fields or behavior**

2. **Enhance `ProcessingStepConfigBase` Directly**
   - Add `_portable_processing_source_dir` private field (Tier 3)
   - Add `portable_processing_source_dir` property
   - Add `portable_effective_source_dir` property
   - Enhance `model_dump()` to include portable paths
   - **No changes to existing fields or behavior**

3. **Update Step Builders (Single Line Change)**
   ```python
   # Current step builder code:
   def _create_estimator(self, output_path=None) -> XGBoost:
       return XGBoost(
           source_dir=self.config.source_dir,  # Uses absolute path
           # ... other parameters
       )
   
   # Enhanced step builder code:
   def _create_estimator(self, output_path=None) -> XGBoost:
       return XGBoost(
           source_dir=self.config.portable_source_dir or self.config.source_dir,  # Uses portable path with fallback
           # ... other parameters
       )
   ```

#### **Phase 2: Automatic Conversion (Zero User Impact)**

1. **Transparent Path Conversion**
   - Users continue to provide absolute paths exactly as before
   - Config classes automatically convert to relative paths internally
   - Serialized JSON contains both original and portable paths
   - **Zero changes to user workflow or existing code**

2. **Complete Backward Compatibility**
   - All existing configurations continue to work without modification
   - All existing code continues to work without changes
   - New portable functionality is additive only
   - Step builders work with both absolute and relative paths

#### **Phase 3: Simple Testing**

1. **Unit Tests for Path Conversion**
   ```python
   def test_portable_path_conversion():
       config = BasePipelineConfig(  # Using existing class directly
           source_dir="/home/user/cursus/dockers/xgboost_atoz",
           # ... other required fields
       )
       
       # Should convert to relative path automatically
       assert config.portable_source_dir == "../../dockers/xgboost_atoz"
       
       # Serialization should include both original and portable paths
       data = config.model_dump()
       assert data["source_dir"] == "/home/user/cursus/dockers/xgboost_atoz"  # Original preserved
       assert data["portable_source_dir"] == "../../dockers/xgboost_atoz"    # Portable added
   
   def test_backward_compatibility():
       """Test that existing functionality is unchanged."""
       config = BasePipelineConfig(
           source_dir="/home/user/cursus/dockers/xgboost_atoz",
           # ... other required fields
       )
       
       # Existing property should work exactly as before
       assert config.source_dir == "/home/user/cursus/dockers/xgboost_atoz"
       
       # All existing methods should work unchanged
       # ... test existing functionality
   
   def test_processing_config_enhancement():
       """Test ProcessingStepConfigBase enhancements."""
       config = ProcessingStepConfigBase(
           processing_source_dir="/home/user/cursus/dockers/xgboost_atoz/scripts",
           # ... other required fields
       )
       
       # New portable functionality
       assert config.portable_processing_source_dir == "../../dockers/xgboost_atoz/scripts"
       
       # Existing functionality unchanged
       assert config.processing_source_dir == "/home/user/cursus/dockers/xgboost_atoz/scripts"
       assert config.effective_source_dir == "/home/user/cursus/dockers/xgboost_atoz/scripts"
   ```

2. **Integration Tests**
   - Test step creation with portable paths
   - Verify paths resolve correctly in different environments
   - Confirm complete backward compatibility
   - Test that existing configurations work without modification

### Implementation Details

#### Workspace Context Detection

```python
class WorkspaceDetector:
    """Detects the current workspace context."""
    
    @staticmethod
    def detect_package_installation() -> Optional[str]:
        """Detect if running from a package installation."""
        try:
            import cursus
            package_path = Path(cursus.__file__).parent
            
            # Check if it's a site-packages installation
            if "site-packages" in str(package_path):
                return "cursus_package"
            
            # Check if it's a development installation
            if (package_path / "setup.py").exists() or (package_path / "pyproject.toml").exists():
                return "cursus_source"
            
        except ImportError:
            pass
        
        return None
    
    @staticmethod
    def detect_workspace_directory() -> Optional[str]:
        """Detect workspace directory from current working directory."""
        cwd = Path.cwd()
        
        # Look for workspace indicators
        workspace_indicators = [
            ".cursus_workspace",
            "development/projects",
            "slipbox",
            "pipeline_config"
        ]
        
        current = cwd
        while current != current.parent:
            for indicator in workspace_indicators:
                if (current / indicator).exists():
                    return f"workspace_{current.name}"
            current = current.parent
        
        return None
```

#### Path Resolution Caching

```python
class PathResolutionCache:
    """Caches resolved paths for performance."""
    
    def __init__(self):
        self._cache: Dict[Tuple[str, str], str] = {}
    
    def get_cached_path(self, relative_path: str, workspace_context: str) -> Optional[str]:
        """Get cached resolved path."""
        cache_key = (relative_path, workspace_context)
        return self._cache.get(cache_key)
    
    def cache_path(self, relative_path: str, workspace_context: str, absolute_path: str) -> None:
        """Cache a resolved path."""
        cache_key = (relative_path, workspace_context)
        self._cache[cache_key] = absolute_path
    
    def clear_cache(self) -> None:
        """Clear the path resolution cache."""
        self._cache.clear()
```

### Configuration File Format

#### Before (Non-Portable)
```json
{
  "configuration": {
    "shared": {
      "source_dir": "/home/ec2-user/SageMaker/Cursus/dockers/xgboost_atoz"
    },
    "specific": {
      "TabularPreprocessing_training": {
        "processing_source_dir": "/home/ec2-user/SageMaker/Cursus/dockers/xgboost_atoz/scripts",
        "processing_entry_point": "tabular_preprocessing.py",
        "script_path": "/home/ec2-user/SageMaker/Cursus/dockers/xgboost_atoz/scripts/tabular_preprocessing.py"
      }
    }
  }
}
```

#### After (Portable - Both Original and Portable Paths)
```json
{
  "configuration": {
    "shared": {
      "source_dir": "/home/ec2-user/SageMaker/Cursus/dockers/xgboost_atoz",
      "portable_source_dir": "../../dockers/xgboost_atoz"
    },
    "specific": {
      "TabularPreprocessing_training": {
        "processing_source_dir": "/home/ec2-user/SageMaker/Cursus/dockers/xgboost_atoz/scripts",
        "portable_processing_source_dir": "../../dockers/xgboost_atoz/scripts",
        "processing_entry_point": "tabular_preprocessing.py"
      }
    }
  },
  "metadata": {
    "portable_format_version": "2.0",
    "path_resolution_method": "step_builder_relative",
    "description": "Both original absolute paths and portable relative paths are preserved"
  }
}
```

**Example Path Resolution**:
- Step Builder Location: `/path/to/cursus/src/cursus/steps/builders/builder_xgboost_training_step.py`
- Relative Path: `../../dockers/xgboost_atoz`
- Resolved Absolute Path: `/path/to/cursus/dockers/xgboost_atoz`

This works consistently whether cursus is:
- Installed via PyPI: `/usr/local/lib/python3.9/site-packages/cursus/...`
- Source installation: `/home/user/projects/cursus/...`
- Container deployment: `/app/cursus/...`

### Benefits and Impact

#### 1. **Universal Portability**
- Configuration files work across all deployment environments
- No manual path adjustments needed when sharing configurations
- Seamless operation in CI/CD pipelines

#### 2. **Workspace Isolation**
- Different projects can maintain separate workspace contexts
- Multi-developer environments supported
- Project-specific configurations possible

#### 3. **Deployment Flexibility**
- PyPI package installations supported
- Source installations supported
- Container deployments supported
- Serverless environments supported

#### 4. **Backward Compatibility**
- Existing absolute path configurations continue to work
- Gradual migration path available
- No breaking changes to existing workflows

#### 5. **Enhanced Developer Experience**
- Automatic workspace detection
- Intelligent path resolution
- Clear error messages for path resolution failures

### Error Handling and Fallbacks

#### Path Resolution Failures
```python
class PathResolutionError(Exception):
    """Raised when path resolution fails."""
    pass

def resolve_path_with_fallbacks(
    relative_path: str, 
    workspace_context: str,
    fallback_paths: Optional[List[str]] = None
) -> str:
    """Resolve path with multiple fallback strategies."""
    
    # Primary resolution
    try:
        workspace_manager = WorkspaceContextManager()
        resolved_path = workspace_manager.resolve_path(relative_path, workspace_context)
        if resolved_path and resolved_path.exists():
            return str(resolved_path)
    except Exception as e:
        logger.warning(f"Primary path resolution failed: {e}")
    
    # Fallback to step catalog discovery
    try:
        catalog = StepCatalog.get_instance()
        discovered_path = catalog.discover_script_path(
            step_name="Unknown",  # Will be provided by caller
            entry_point=Path(relative_path).name,
            workspace_context=workspace_context
        )
        if discovered_path:
            return discovered_path
    except Exception as e:
        logger.warning(f"Step catalog discovery failed: {e}")
    
    # Fallback to provided fallback paths
    if fallback_paths:
        for fallback_path in fallback_paths:
            if Path(fallback_path).exists():
                logger.info(f"Using fallback path: {fallback_path}")
                return fallback_path
    
    # Final fallback: return relative path as-is (may work in some contexts)
    logger.warning(f"All path resolution strategies failed, returning relative path: {relative_path}")
    return relative_path
```

### Testing Strategy

#### Unit Tests
```python
class TestPortablePathResolution(unittest.TestCase):
    
    def setUp(self):
        self.workspace_manager = WorkspaceContextManager()
        self.temp_workspace = Path(tempfile.mkdtemp())
        self.workspace_manager.register_workspace("test_workspace", self.temp_workspace)
    
    def test_relative_path_resolution(self):
        """Test basic relative path resolution."""
        # Create test structure
        script_dir = self.temp_workspace / "scripts"
        script_dir.mkdir()
        test_script = script_dir / "test_script.py"
        test_script.write_text("# Test script")
        
        # Test resolution
        resolved_path = self.workspace_manager.resolve_path("scripts/test_script.py", "test_workspace")
        self.assertEqual(resolved_path, test_script)
        self.assertTrue(resolved_path.exists())
    
    def test_config_serialization_deserialization(self):
        """Test portable config serialization and deserialization."""
        # Create config with absolute paths
        config_data = {
            "workspace_context": "test_workspace",
            "source_dir": str(self.temp_workspace / "dockers" / "test"),
            "processing_entry_point": "test_script.py"
        }
        
        config = PortableProcessingStepConfigBase(**config_data)
        
        # Serialize (should convert to relative paths)
        serialized = PortableConfigSerializer.serialize_config(config)
        self.assertEqual(serialized["source_dir"], "dockers/test")
        
        # Deserialize (should resolve back to absolute paths)
        deserialized = PortableConfigSerializer.deserialize_config(
            serialized, PortableProcessingStepConfigBase
        )
        self.assertEqual(deserialized.get_absolute_source_dir(), str(self.temp_workspace / "dockers" / "test"))
```

#### Integration Tests
```python
class TestCrossEnvironmentPortability(unittest.TestCase):
    
    def test_package_installation_simulation(self):
        """Test portability in simulated package installation."""
        # Simulate package installation structure
        package_root = Path(tempfile.mkdtemp()) / "site-packages" / "cursus"
        package_root.mkdir(parents=True)
        
        # Create test configuration
        config_data = {
            "workspace_context": "cursus_package",
            "source_dir": "dockers/xgboost_atoz",
            "processing_entry_point": "training.py"
        }
        
        # Test that configuration works in package context
        workspace_manager = WorkspaceContextManager()
        workspace_manager.register_workspace("cursus_package", package_root)
        
        config = PortableProcessingStepConfigBase(**config_data)
        expected_path = package_root / "dockers" / "xgboost_atoz"
        self.assertEqual(config.get_absolute_source_dir(), str(expected_path))
    
    def test_source_installation_simulation(self):
        """Test portability in simulated source installation."""
        # Similar test for source installation context
        pass
```

## Conclusion

The Configuration Portability Path Resolution Design provides a comprehensive solution to the critical portability issues in the cursus configuration system. By implementing workspace-aware relative path resolution, the system achieves:

### **Technical Excellence**
- **Universal Portability**: Configurations work across all deployment environments
- **Intelligent Resolution**: Automatic path discovery and resolution
- **Backward Compatibility**: Existing configurations continue to work
- **Performance Optimization**: Cached path resolution for efficiency

### **Operational Benefits**
- **Seamless Sharing**: Configurations can be shared between developers and systems
- **CI/CD Integration**: Automated pipelines can use saved configurations
- **Deployment Flexibility**: Support for all deployment scenarios
- **Reduced Maintenance**: No manual path adjustments needed

### **Strategic Impact**
The design transforms cursus from a deployment-fragile system into a truly portable framework, enabling the universal machine learning pipeline orchestration vision. This foundation supports continued innovation while maintaining the reliability and usability that developers require.

### **Implementation Roadmap**
1. **Phase 1**: Enhanced configuration classes with step builder-relative path resolution
2. **Phase 2**: Integration with step catalog and automatic path discovery
3. **Phase 3**: Step builder integration and automatic path conversion
4. **Phase 4**: Migration utilities and comprehensive testing
5. **Phase 5**: Documentation and developer training

### **Key Innovation: Step Builder-Relative Approach**
The core innovation of this design is using the **step builder file location** as the reference point for path resolution. This approach:

- **Eliminates Configuration Complexity**: No need to configure workspace contexts
- **Provides Automatic Portability**: Paths are automatically portable across all environments
- **Maintains Intuitive Behavior**: Relative paths work as developers expect them to
- **Enables Seamless Sharing**: Configuration files work immediately on any system

**Example Workflow**:
1. Developer creates config with absolute path: `/home/user/cursus/dockers/xgboost_atoz`
2. System automatically converts to relative path: `../../dockers/xgboost_atoz` (relative to step builder)
3. Config saved with relative path works on any system where cursus is installed
4. Step builder automatically resolves relative path to correct absolute path at runtime

This design positions cursus as a leading portable machine learning framework capable of seamless operation across any environment while maintaining the simplicity and power that makes it effective for ML pipeline development.

## References

### Related Design Documents
- [Cursus Package Portability Architecture Design](./cursus_package_portability_architecture_design.md) - Overall portability architecture
- [Unified Step Catalog System Design](./unified_step_catalog_system_design.md) - Step catalog integration
- [Workspace Aware Configuration Design](./workspace_aware_configuration_design.md) - Workspace awareness patterns

### Implementation Files
- `src/cursus/core/base/config_base.py` - Base configuration classes
- `src/cursus/steps/configs/config_processing_step_base.py` - Processing step configuration
- `src/cursus/step_catalog/step_catalog.py` - Step catalog system
- `src/cursus/core/workspace/` - Workspace management components

### Standards and Guidelines
- [Documentation YAML Frontmatter Standard](./documentation_yaml_frontmatter_standard.md) - Documentation formatting standards

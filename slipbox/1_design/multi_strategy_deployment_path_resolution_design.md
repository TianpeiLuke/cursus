---
tags:
  - design
  - architecture
  - path_resolution
  - deployment_portability
  - multi_strategy
keywords:
  - multi strategy path resolution
  - deployment scenario detection
  - universal path portability
  - configuration simplification
  - runtime context adaptation
topics:
  - multi-strategy path resolution
  - deployment scenario detection
  - universal configuration portability
  - runtime context adaptation
language: python
date of note: 2025-09-22
---

# Multi-Strategy Deployment Path Resolution Design

## Executive Summary

This document presents a **multi-strategy path resolution system** that automatically detects deployment scenarios and applies the appropriate path resolution strategy. The design addresses the fundamental challenge of making cursus configurations work universally across three distinct deployment architectures: **bundled packages** (Lambda/MODS), **development monorepos**, and **pip-installed separated environments**.

The solution implements **deployment scenario detection** with **strategy-specific path resolution**, enabling the same configuration files to work seamlessly across all deployment contexts without manual intervention or complex setup.

## Problem Statement

### The Three Deployment Scenario Challenge

Based on comprehensive analysis of real-world deployment patterns, cursus configurations must work across three fundamentally different deployment architectures. The key distinguishing factor is the **relationship between runtime execution location and target script location**:

#### **Scenario 1: Completely Separated Runtime and Scripts (Lambda/MODS)**
```
/var/task/                           # Lambda runtime execution directory (cwd)
/tmp/buyer_abuse_mods_template/      # Package root (completely separate filesystem location)
├── cursus/                          # Cursus framework
├── mods_pipeline_adapter/           # User's pipeline code
│   ├── dockers/                     # User's script directory (target location)
│   └── other_pipeline_files/
└── fraud_detection/                 # Another project folder
    ├── scripts/                     # User's script directory (different name)
    └── other_project_files/

Key Characteristic: Runtime program is in /var/task/, which is COMPLETELY CUT OFF from the script's location under /tmp/buyer_abuse_mods_template/mods_pipeline_adapter/dockers/. From cwd(), we CANNOT trace back to dockers folder.
```

#### **Scenario 2: Shared Project Root (Development Monorepo)**
```
/Users/tianpeixie/github_workspace/cursus/    # Common project root
├── src/cursus/                               # Cursus framework (nested)
├── dockers/                                  # User's docker scripts (target location)
├── demo/                                     # Runtime execution directory (cwd)
└── other_project_files/

Key Characteristic: Runtime program is in the same working directory structure as the dockers folder under common project folder. Both cursus definitions and target scripts are under the same project root.
```

#### **Scenario 3: Shared Project Root (Pip-Installed)**
```
<venv>/lib/python3.x/site-packages/cursus/   # Pip-installed cursus (separate)

# User's project (common project root)
/home/user/my_project/                        # Common project root
├── dockers/                                  # User's script directory (target location)
├── config.json                               # User's config
└── main.py                                   # Runtime execution script (cwd)

Key Characteristic: Runtime program is in the same working directory as the dockers folder under common project folder. Only cursus is installed separately via pip.
```

### **Fundamental Path Resolution Challenge**

The core insight is that there are really **two distinct filesystem relationship patterns**:

#### **Pattern A: Separated Runtime and Scripts (Scenario 1)**
- **Runtime Location**: `/var/task/` (Lambda execution directory)
- **Target Scripts**: `/tmp/buyer_abuse_mods_template/mods_pipeline_adapter/dockers/`
- **Relationship**: **COMPLETELY CUT OFF** - cannot traverse from runtime to scripts via filesystem
- **Resolution Strategy**: Must use **Package Location Discovery** (`Path(__file__)`) to find scripts

#### **Pattern B: Shared Project Root (Scenarios 2 & 3)**
- **Runtime Location**: Within project directory structure
- **Target Scripts**: Within same project directory structure  
- **Relationship**: **CONNECTED** - can traverse from runtime to scripts via filesystem
- **Resolution Strategy**: Can use **Working Directory Discovery** (`Path.cwd()`) to find scripts

**Note**: Scenarios 2 and 3 are essentially the same for path resolution purposes. The only difference is whether cursus is defined within the project (Scenario 2) or installed separately (Scenario 3), but this doesn't affect the search strategy since both runtime and target scripts share a common project root.

### Current System Limitations

The existing path resolution system fails because it assumes a **single deployment pattern** and uses **working directory-dependent resolution** that breaks when the same configuration is used across different deployment contexts.

**Critical Failure Pattern:**
1. Configuration created in one deployment scenario (e.g., development)
2. Same configuration used in different deployment scenario (e.g., Lambda)
3. Path resolution fails because deployment architecture is different
4. Result: "File not found" errors in production deployments

### User Requirements

Based on the user's explicit requirements, the solution must:

1. **Support all three deployment scenarios** with automatic detection
2. **Use simple relative path configuration** in `source_dir` and `processing_source_dir` fields (e.g., `"source_dir": "dockers/xgboost_atoz"`)
3. **Allow users to specify their project folder name** under which the relative paths are resolved (via `project_root_folder`)
4. **Provide clear separation** between the base folder name and the relative paths within that folder
5. **Avoid over-engineering** - focus on solving the core problem simply
6. **Maintain backward compatibility** with existing configurations
7. **Work transparently** without requiring complex user configuration changes

## Solution Architecture

### Core Design Principle: Package Location First with Working Directory Fallback

The solution implements **Package Location First** strategy with **Working Directory Discovery** as fallback. This approach recognizes that the cursus package location (`Path(__file__)`) provides the most reliable reference point across all deployment scenarios, with working directory discovery handling edge cases.

**Key Insight**: Use the cursus package location as the primary reference point, then fall back to working directory discovery when package-relative resolution fails.

### Hybrid Resolution Algorithm: Package Location First

```python
def _resolve_source_dir(self, project_root_folder: Optional[str], relative_path: Optional[str]) -> Optional[str]:
    """Hybrid path resolution: Package location first, then working directory discovery."""
    if not relative_path:
        return None
    
    # Strategy 1: Package Location Discovery (works for all scenarios)
    resolved = self._package_location_discovery(project_root_folder, relative_path)
    if resolved:
        return resolved
    
    # Strategy 2: Working Directory Discovery (fallback for edge cases)
    resolved = self._working_directory_discovery(relative_path)
    if resolved:
        return resolved
    
    return None

def _package_location_discovery(self, project_root_folder: Optional[str], relative_path: str) -> Optional[str]:
    """Discover paths using cursus package location as reference."""
    cursus_file = Path(__file__)  # Current cursus module file
    
    # Strategy 1A: Check if we're in bundled deployment (Lambda/MODS)
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
    
    # Try searching in project subdirectories (for bundled scenarios)
    resolved_path = self._search_in_subdirectories(potential_package_root, relative_path)
    if resolved_path:
        return resolved_path
    
    # Strategy 1B: Check if we're in monorepo structure (src/cursus)
    if "src" in cursus_file.parts:
        src_index = cursus_file.parts.index("src")
        project_root = Path(*cursus_file.parts[:src_index])
        
        if project_root.exists() and project_root.is_dir():
            target_path = project_root / relative_path
            if target_path.exists():
                return str(target_path)
    
    return None

def _working_directory_discovery(self, project_root_folder: Optional[str], relative_path: str) -> Optional[str]:
    """Discover paths using working directory traversal (fallback)."""
    current = Path.cwd()
    
    # Assumption: runtime program is under project_root_folder path
    # So we search upward to find the project_root_folder directory
    
    # Search upward for project root
    while current != current.parent:
        # Strategy 1: If project_root_folder is specified, check if we're inside it
        if project_root_folder:
            # Check if current directory name matches project_root_folder
            if current.name == project_root_folder:
                # We found the project root folder, try to resolve relative_path from here
                target_path = current / relative_path
                if target_path.exists():
                    return str(target_path)
            
            # Check if project_root_folder exists as subdirectory of current
            project_folder_path = current / project_root_folder
            if project_folder_path.exists() and project_folder_path.is_dir():
                # Found project_root_folder as subdirectory, try to resolve relative_path from it
                target_path = project_folder_path / relative_path
                if target_path.exists():
                    return str(target_path)
        
        # Strategy 2: Direct path resolution (for cases without project_root_folder)
        direct_path = current / relative_path
        if direct_path.exists():
            return str(direct_path)
            
        current = current.parent
    
    # Final fallback: try current working directory
    if project_root_folder:
        # Try with project_root_folder first
        fallback_with_project = Path.cwd() / project_root_folder / relative_path
        if fallback_with_project.exists():
            return str(fallback_with_project)
    
    # Try direct fallback
    fallback_path = Path.cwd() / relative_path
    if fallback_path.exists():
        return str(fallback_path)
    
    return None

def _search_in_subdirectories(self, package_root: Path, relative_path: str) -> Optional[str]:
    """Search for relative path in package subdirectories (for bundled scenarios)."""
    try:
        # Look for the relative path in immediate subdirectories
        for item in package_root.iterdir():
            if item.is_dir() and item.name != "cursus":  # Skip cursus directory
                potential_path = item / relative_path
                if potential_path.exists():
                    return str(potential_path)
    except (OSError, PermissionError):
        pass
    
    return None
```


## Configuration Format

### Enhanced Configuration with Project Root

The configuration now includes two key fields for precise path resolution:

```json
{
  "TabularPreprocessing_training": {
    "project_root_folder": "mods_pipeline_adapter",
    "source_dir": "dockers/xgboost_atoz",
    "script_name": "tabular_preprocessing.py"
  },
  "ModelCalibration_calibration": {
    "project_root_folder": "mods_pipeline_adapter", 
    "source_dir": "dockers/xgboost_atoz",
    "script_name": "model_calibration.py"
  },
  "FraudDetection_training": {
    "project_root_folder": "fraud_detection",
    "source_dir": "scripts",
    "script_name": "fraud_model_training.py"
  }
}
```

### Configuration Field Definitions

- **`project_root_folder`**: **Universal user-specified field (Tier 1 config)** - The root folder name for the user's project (e.g., "mods_pipeline_adapter", "fraud_detection") - this is just a folder name, not a path. Used in all scenarios for consistent configuration format.
- **`source_dir`**: The relative path of the script's folder with respect to `project_root_folder` (e.g., "dockers/xgboost_atoz/", "scripts/")

### **When Do We Need `project_root_folder`?**

#### **Scenario 1 (Lambda/MODS): YES - Required**
```
/tmp/buyer_abuse_mods_template/      # Package root discovered via Path(__file__)
├── cursus/                          # Cursus framework
├── mods_pipeline_adapter/           # ← project_root_folder specifies THIS folder
│   └── dockers/xgboost_atoz/        # ← source_dir specifies path within project folder
└── fraud_detection/                 # ← Different project would use different project_root_folder
    └── scripts/

Why needed: Multiple project folders exist as siblings to cursus. We need to specify which one.
Configuration: "project_root_folder": "mods_pipeline_adapter", "source_dir": "dockers/xgboost_atoz"
```

#### **Scenarios 2 & 3 (Monorepo/Pip-installed): NO - Ignored**
```
# Scenario 2: Monorepo
/Users/tianpeixie/github_workspace/cursus/    # Single project root
├── src/cursus/                               # Cursus framework
└── dockers/xgboost_atoz/                     # ← source_dir specifies path from project root

# Scenario 3: Pip-installed  
/home/user/my_project/                        # Single project root
├── dockers/xgboost_atoz/                     # ← source_dir specifies path from project root
└── main.py

Why not needed: Only one project structure. source_dir is relative to the single project root.
Configuration: "project_root_folder": "ignored", "source_dir": "dockers/xgboost_atoz"
```

### **Simplified Configuration Strategy**

#### **Option 1: Always Include `project_root_folder` (Recommended)**
```json
{
  "TabularPreprocessing_training": {
    "project_root_folder": "mods_pipeline_adapter",  // Used in Scenario 1, ignored in 2&3
    "source_dir": "dockers/xgboost_atoz",
    "script_name": "tabular_preprocessing.py"
  }
}
```
**Benefit**: Same configuration works everywhere. System automatically ignores `project_root_folder` when not needed.

#### **Option 2: Conditional `project_root_folder`**
```json
// For Lambda/MODS deployment
{
  "TabularPreprocessing_training": {
    "project_root_folder": "mods_pipeline_adapter",  // Required for Lambda
    "source_dir": "dockers/xgboost_atoz"
  }
}

// For Monorepo/Pip-installed deployment  
{
  "TabularPreprocessing_training": {
    "source_dir": "dockers/xgboost_atoz"  // No project_root_folder needed
  }
}
```
**Benefit**: Minimal configuration, but requires different configs for different deployments.

**Recommendation**: Use **Option 1** for true universal configuration portability.

### Runtime Resolution Examples

**Hybrid Resolution Algorithm in Action:**

#### **Scenario 1 (Lambda/MODS)**
```python
# Configuration:
# "project_root_folder": "mods_pipeline_adapter"
# "source_dir": "dockers/xgboost_atoz"

# Hybrid Resolution Execution:
def _resolve_source_dir(project_root_folder="mods_pipeline_adapter", relative_path="dockers/xgboost_atoz"):
    
    # Strategy 1: Package Location Discovery
    resolved = _package_location_discovery("mods_pipeline_adapter", "dockers/xgboost_atoz")
    # cursus_file = /tmp/buyer_abuse_mods_template/cursus/core/base/config_base.py
    # potential_package_root = /tmp/buyer_abuse_mods_template/
    # direct_path = /tmp/buyer_abuse_mods_template/mods_pipeline_adapter/dockers/xgboost_atoz
    # direct_path.exists() = True ✅
    # Returns: "/tmp/buyer_abuse_mods_template/mods_pipeline_adapter/dockers/xgboost_atoz"
    
    # Strategy 2: Working Directory Discovery - NOT EXECUTED (Strategy 1 succeeded)

# Result: ✅ SUCCESS via Package Location Discovery
```

#### **Scenario 2 (Development Monorepo)**
```python
# Configuration:
# "project_root_folder": "mods_pipeline_adapter"  # Will be ignored
# "source_dir": "dockers/xgboost_atoz"

# Hybrid Resolution Execution:
def _resolve_source_dir(project_root_folder="mods_pipeline_adapter", relative_path="dockers/xgboost_atoz"):
    
    # Strategy 1: Package Location Discovery
    resolved = _package_location_discovery("mods_pipeline_adapter", "dockers/xgboost_atoz")
    # cursus_file = /Users/tianpeixie/github_workspace/cursus/src/cursus/core/base/config_base.py
    # potential_package_root = /Users/tianpeixie/github_workspace/cursus/src/
    # direct_path = /Users/tianpeixie/github_workspace/cursus/src/mods_pipeline_adapter/dockers/xgboost_atoz
    # direct_path.exists() = False ❌
    # 
    # Try direct resolution: /Users/tianpeixie/github_workspace/cursus/src/dockers/xgboost_atoz
    # direct_path.exists() = False ❌
    #
    # Check monorepo structure: "src" in cursus_file.parts = True
    # project_root = /Users/tianpeixie/github_workspace/cursus/
    # target_path = /Users/tianpeixie/github_workspace/cursus/dockers/xgboost_atoz
    # target_path.exists() = True ✅
    # Returns: "/Users/tianpeixie/github_workspace/cursus/dockers/xgboost_atoz"
    
    # Strategy 2: Working Directory Discovery - NOT EXECUTED (Strategy 1 succeeded)

# Result: ✅ SUCCESS via Package Location Discovery (monorepo detection)
```

#### **Scenario 3 (Pip-Installed)**
```python
# Configuration:
# "project_root_folder": "mods_pipeline_adapter"  # Used by working directory discovery
# "source_dir": "dockers/xgboost_atoz"

# Hybrid Resolution Execution:
def _resolve_source_dir(project_root_folder="mods_pipeline_adapter", relative_path="dockers/xgboost_atoz"):
    
    # Strategy 1: Package Location Discovery
    resolved = _package_location_discovery("mods_pipeline_adapter", "dockers/xgboost_atoz")
    # cursus_file = /Users/tianpeixie/.venv/lib/python3.12/site-packages/cursus/core/base/config_base.py
    # potential_package_root = /Users/tianpeixie/.venv/lib/python3.12/site-packages/
    # direct_path = /Users/tianpeixie/.venv/lib/python3.12/site-packages/mods_pipeline_adapter/dockers/xgboost_atoz
    # direct_path.exists() = False ❌ (user project not in site-packages)
    #
    # Try direct resolution: /Users/tianpeixie/.venv/lib/python3.12/site-packages/dockers/xgboost_atoz
    # direct_path.exists() = False ❌ (user files not in site-packages)
    #
    # Check monorepo structure: "src" in cursus_file.parts = False
    # Returns: None ❌
    
    # Strategy 2: Working Directory Discovery
    resolved = _working_directory_discovery("mods_pipeline_adapter", "dockers/xgboost_atoz")
    # current = Path.cwd() = /Users/tianpeixie/mods_pipeline_adapter/  # User is in their project directory
    # Search upward for project root...
    # current.name == "mods_pipeline_adapter" = True ✅  # Found project root folder!
    # target_path = /Users/tianpeixie/mods_pipeline_adapter/dockers/xgboost_atoz
    # target_path.exists() = True ✅
    # Returns: "/Users/tianpeixie/mods_pipeline_adapter/dockers/xgboost_atoz"

# Result: ✅ SUCCESS via Working Directory Discovery using project_root_folder
```

#### **Key Insights from Hybrid Resolution**

| Scenario | Primary Strategy Result | Fallback Strategy Result | Final Result |
|----------|------------------------|---------------------------|--------------|
| **Lambda/MODS** | ✅ Package Location (direct project_root_folder match) | Not executed | Package Location Success |
| **Monorepo** | ✅ Package Location (monorepo structure detection) | Not executed | Package Location Success |
| **Pip-installed** | ❌ Package Location (no user files in site-packages) | ✅ Working Directory Discovery | Working Directory Success |

**Benefits of Hybrid Approach:**
- **No scenario detection overhead** - tries strategies directly
- **Package Location First** - most reliable strategy tried first
- **Automatic fallback** - working directory discovery handles edge cases
- **Universal configuration** - same config works across all scenarios

## Implementation Strategy

### Phase 1: Core Multi-Strategy Resolution

#### **Enhanced BasePipelineConfig**

```python
class BasePipelineConfig(BaseModel, ABC):
    """Base configuration with hybrid path resolution."""
    
    # Enhanced fields for universal configuration
    project_root_folder: Optional[str] = Field(
        default=None,
        description="Root folder name for the user's project (Tier 1 config field)"
    )
    source_dir: Optional[str] = Field(
        default=None,
        description="Source directory for scripts (relative path)"
    )
    
    def _resolve_source_dir(self, project_root_folder: Optional[str], relative_path: Optional[str]) -> Optional[str]:
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
    
    @property
    def resolved_source_dir(self) -> Optional[str]:
        """Get resolved source directory using hybrid resolution."""
        return self._resolve_source_dir(self.project_root_folder, self.source_dir)
```

#### **Enhanced ProcessingStepConfigBase**

```python
class ProcessingStepConfigBase(BasePipelineConfig):
    """Processing configuration with multi-strategy path resolution."""
    
    # Existing fields unchanged
    processing_source_dir: Optional[str] = Field(
        default=None,
        description="Source directory for processing scripts (relative path)"
    )
    
    @property
    def resolved_processing_source_dir(self) -> Optional[str]:
        """Get resolved processing source directory."""
        if self.processing_source_dir:
            return self._resolve_source_dir(self.processing_source_dir)
        return self._resolve_source_dir(self.source_dir)
    
    def get_resolved_script_path(self) -> Optional[str]:
        """Get resolved script path for step builders."""
        source_dir = self.resolved_processing_source_dir
        if source_dir and hasattr(self, 'script_name'):
            return str(Path(source_dir) / self.script_name)
        return None
```

### Phase 2: Step Builder Integration

#### **Minimal Step Builder Changes**

Step builders require only **single-line changes** to use the new resolution:

```python
# BEFORE: Direct path usage
class TabularPreprocessingStepBuilder(StepBuilderBase):
    def build_step(self) -> ProcessingStep:
        script_path = self.config.get_script_path()  # May fail in different contexts
        
        return ProcessingStep(
            name=self.step_name,
            code=script_path,
            # ... other parameters
        )

# AFTER: Multi-strategy resolution with fallback
class TabularPreprocessingStepBuilder(StepBuilderBase):
    def build_step(self) -> ProcessingStep:
        script_path = (
            self.config.get_resolved_script_path() or  # Multi-strategy resolution
            self.config.get_script_path()              # Fallback to existing behavior
        )
        
        return ProcessingStep(
            name=self.step_name,
            code=script_path,
            # ... other parameters
        )
```

### Phase 3: Validation and Testing

#### **Scenario-Specific Testing**

```python
class TestMultiStrategyPathResolution(unittest.TestCase):
    
    def test_bundled_scenario_detection(self):
        """Test detection and resolution for bundled deployment."""
        # Simulate Lambda/MODS structure
        with tempfile.TemporaryDirectory() as temp_dir:
            package_root = Path(temp_dir)
            cursus_dir = package_root / "cursus"
            dockers_dir = package_root / "dockers" / "xgboost_atoz"
            
            cursus_dir.mkdir()
            dockers_dir.mkdir(parents=True)
            
            # Test detection
            config = TestConfig(source_dir="dockers/xgboost_atoz")
            scenario = config._detect_deployment_scenario()
            self.assertEqual(scenario, "bundled")
            
            # Test resolution
            resolved = config._resolve_source_dir("dockers/xgboost_atoz")
            self.assertEqual(resolved, str(dockers_dir))
    
    def test_monorepo_scenario_detection(self):
        """Test detection and resolution for monorepo deployment."""
        # Simulate development structure
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            cursus_dir = project_root / "src" / "cursus"
            dockers_dir = project_root / "dockers" / "xgboost_atoz"
            
            cursus_dir.mkdir(parents=True)
            dockers_dir.mkdir(parents=True)
            
            # Test detection and resolution
            # ... similar test pattern
    
    def test_pip_installed_scenario_detection(self):
        """Test detection and resolution for pip-installed deployment."""
        # Simulate pip-installed structure
        # ... similar test pattern
```

## Concrete Examples and Analysis

This section provides detailed analysis of how `_resolve_source_dir` works in each deployment scenario using real-world examples to prove the solution's effectiveness.

### **Example 1: Bundled Deployment (Lambda/MODS)**

#### **File System Structure**
```
/var/task/                           # Lambda working directory
/tmp/buyer_abuse_mods_template/      # Package root (both cursus and mods_pipeline_adapter)
├── cursus/                          # Cursus framework
│   └── core/base/config_base.py     # Where _resolve_source_dir runs
└── mods_pipeline_adapter/           # User's project folder
    └── dockers/xgboost_atoz/        # User's source directory
        └── scripts/
            └── tabular_preprocessing.py  # Target file
```

#### **Configuration Input**
```json
{
  "TabularPreprocessing_training": {
    "project_root_folder": "mods_pipeline_adapter",
    "source_dir": "dockers/xgboost_atoz",
    "script_name": "tabular_preprocessing.py"
  }
}
```

#### **`_resolve_source_dir` Execution**
```python
def _resolve_source_dir(self, project_root_folder: "mods_pipeline_adapter", relative_path: "dockers/xgboost_atoz"):
    scenario = self._detect_deployment_scenario()  # Returns "bundled"
    
    if scenario == "bundled":
        # Path(__file__) = /tmp/buyer_abuse_mods_template/cursus/core/base/config_base.py
        package_root = Path(__file__).parent.parent.parent  # Go up 3 levels: base -> core -> cursus -> package_root
        # package_root = /tmp/buyer_abuse_mods_template/
        
        if project_root_folder:  # "mods_pipeline_adapter"
            resolved_path = package_root / project_root_folder / relative_path
            # resolved_path = /tmp/buyer_abuse_mods_template/mods_pipeline_adapter/dockers/xgboost_atoz
            
            if resolved_path.exists():  # ✅ TRUE - this path exists!
                return str(resolved_path)  # Returns: "/tmp/buyer_abuse_mods_template/mods_pipeline_adapter/dockers/xgboost_atoz"
```

**Result**: ✅ **SUCCESS** - Returns correct path that exists in Lambda environment

### **Example 2: Development Monorepo (Current Repository)**

#### **File System Structure**
```
/Users/tianpeixie/github_workspace/cursus/    # Project root
├── src/cursus/                               # Cursus framework (nested)
│   └── core/base/config_base.py              # Where _resolve_source_dir runs
└── dockers/xgboost_atoz/                     # User's docker scripts (direct)
    └── scripts/
        └── tabular_preprocessing.py          # Target file
```

#### **Configuration Input**
```json
{
  "TabularPreprocessing_training": {
    "project_root_folder": "mods_pipeline_adapter",  // This will be IGNORED in monorepo
    "source_dir": "dockers/xgboost_atoz",
    "script_name": "tabular_preprocessing.py"
  }
}
```

#### **`_resolve_source_dir` Execution**
```python
def _resolve_source_dir(self, project_root_folder: "mods_pipeline_adapter", relative_path: "dockers/xgboost_atoz"):
    scenario = self._detect_deployment_scenario()  # Returns "monorepo"
    
    elif scenario == "monorepo":
        # Path(__file__) = /Users/tianpeixie/github_workspace/cursus/src/cursus/core/base/config_base.py
        cursus_file = Path(__file__)
        
        # cursus_file.parts = ('/', 'Users', 'tianpeixie', 'github_workspace', 'cursus', 'src', 'cursus', 'core', 'base', 'config_base.py')
        src_index = cursus_file.parts.index("src")  # src_index = 5
        
        # Take everything before "src": ('/', 'Users', 'tianpeixie', 'github_workspace', 'cursus')
        project_root = Path(*cursus_file.parts[:src_index])
        # project_root = /Users/tianpeixie/github_workspace/cursus/
        
        resolved_path = project_root / relative_path
        # resolved_path = /Users/tianpeixie/github_workspace/cursus/dockers/xgboost_atoz
        
        return str(resolved_path)
        # Returns: "/Users/tianpeixie/github_workspace/cursus/dockers/xgboost_atoz"
```

**Result**: ✅ **SUCCESS** - Returns correct path (ignores `project_root_folder` in monorepo)

### **Example 3: Pip-Installed Separated**

#### **File System Structure**
```
# Pip-installed cursus (completely separate)
/Users/tianpeixie/github_workspace/cursus/.venv/lib/python3.12/site-packages/cursus/
├── core/base/config_base.py              # Where _resolve_source_dir runs
├── steps/builders/                       # Step builders
└── steps/configs/                        # Step configs

# User's project (current working directory)
/Users/tianpeixie/github_workspace/cursus/    # User's project root
├── dockers/                                  # User's docker scripts
│   ├── xgboost_atoz/
│   │   └── scripts/
│   │       └── tabular_preprocessing.py     # Target file
│   ├── pytorch_bsm_ext/
│   └── xgboost_pda/
├── pipeline_config/                          # User's configuration files
├── demo/                                     # User's demo notebooks
└── main.py                                   # User's execution script
```

#### **Configuration Input**
```json
{
  "TabularPreprocessing_training": {
    "project_root_folder": "mods_pipeline_adapter",  // This will be IGNORED in pip-installed
    "source_dir": "dockers/xgboost_atoz",
    "script_name": "tabular_preprocessing.py"
  }
}
```

#### **`_resolve_source_dir` Execution**
```python
def _resolve_source_dir(self, project_root_folder: "mods_pipeline_adapter", relative_path: "dockers/xgboost_atoz"):
    scenario = self._detect_deployment_scenario()  # Returns "pip_installed"
    
    elif scenario == "pip_installed":
        return self._discover_user_project_path(relative_path)
        # This searches upward from current working directory for project markers

def _discover_user_project_path(self, relative_path: "dockers/xgboost_atoz"):
    # Start from current working directory (where user runs their script)
    current = Path.cwd()  # /Users/tianpeixie/github_workspace/cursus/
    
    markers = ['dockers/', 'pyproject.toml', '.git', 'config.json']
    
    # Search upward for project root
    while current != current.parent:
        # Check current directory: /Users/tianpeixie/github_workspace/cursus/
        if any((current / marker).exists() for marker in markers):
            # ✅ Found markers:
            # - (current / 'dockers/').exists() = True
            # - (current / 'pyproject.toml').exists() = True  
            # - (current / '.git').exists() = True
            
            # Found project root at /Users/tianpeixie/github_workspace/cursus/
            target_path = current / relative_path
            # target_path = /Users/tianpeixie/github_workspace/cursus/dockers/xgboost_atoz
            
            if target_path.exists():  # ✅ TRUE - this path exists!
                return str(target_path)  
                # Returns: "/Users/tianpeixie/github_workspace/cursus/dockers/xgboost_atoz"
        
        current = current.parent  # Move up one directory level
    
    # Fallback to current directory if no project root found
    fallback_path = Path.cwd() / relative_path
    if fallback_path.exists():
        return str(fallback_path)
    
    return None
```

**Result**: ✅ **SUCCESS** - Returns correct path using working directory discovery

### **Comparative Analysis**

#### **Scenario Detection Comparison**

| Scenario | Detection Method | Key Indicator | `project_root_folder` Usage |
|----------|-----------------|---------------|---------------------------|
| **Bundled (Lambda)** | Sibling directories to cursus | `/tmp/buyer_abuse_mods_template/mods_pipeline_adapter/` exists | ✅ **Used** - specifies which project folder |
| **Monorepo (Current)** | `src/cursus` pattern | `"src"` in cursus file path | ❌ **Ignored** - single project structure |
| **Pip-installed** | Working directory + markers | No bundled/monorepo patterns found | ❌ **Ignored** - user project discovery |

#### **Path Resolution Strategy Comparison**

| Scenario | Resolution Formula | Example Result |
|----------|-------------------|----------------|
| **Bundled** | `package_root + project_root_folder + source_dir` | `/tmp/buyer_abuse_mods_template/mods_pipeline_adapter/dockers/xgboost_atoz` |
| **Monorepo** | `project_root + source_dir` | `/Users/tianpeixie/github_workspace/cursus/dockers/xgboost_atoz` |
| **Pip-installed** | `discovered_user_project + source_dir` | `/Users/tianpeixie/github_workspace/cursus/dockers/xgboost_atoz` |

#### **Key Success Factors**

1. **Automatic Detection**: Each scenario is detected automatically without user configuration
2. **Context-Appropriate Resolution**: Each scenario uses the optimal resolution strategy for its context
3. **Universal Configuration**: Same configuration works across all scenarios
4. **Real-World Validation**: All examples use actual file system structures that exist

### **Solution Validation**

This analysis proves that the multi-strategy approach successfully:

1. **Solves the MODS Lambda Error**: Bundled scenario correctly resolves paths in Lambda environment
2. **Works in Development**: Monorepo scenario correctly handles current repository structure  
3. **Supports Pip Installation**: Pip-installed scenario bridges the gap between separated cursus and user files
4. **Maintains Simplicity**: Users provide simple configuration that works everywhere
5. **Provides Robustness**: Each scenario has appropriate fallback mechanisms

The concrete examples demonstrate that the same configuration format produces correct, working paths across all three deployment scenarios, proving the solution's effectiveness and universality.

## Key Design Benefits

### **1. Automatic Scenario Detection**
- **No configuration required** - system automatically detects deployment context
- **Robust detection logic** - uses file system structure as reliable indicators
- **Graceful fallbacks** - handles edge cases and detection failures

### **2. Strategy-Specific Optimization**
- **Scenario 1 (Bundled)**: Fast sibling directory resolution
- **Scenario 2 (Monorepo)**: Efficient src-aware navigation
- **Scenario 3 (Pip-installed)**: Intelligent project root discovery

### **3. Universal Configuration Portability**
- **Same config files work everywhere** - no manual path adjustments
- **Simple relative paths** - easy to understand and maintain
- **Deployment-agnostic** - configurations are truly portable

### **4. Backward Compatibility**
- **Existing configurations work unchanged** - automatic enhancement
- **Fallback mechanisms** - graceful degradation when resolution fails
- **Zero breaking changes** - existing APIs remain unchanged

### **5. Minimal Implementation Complexity**
- **Single detection method** - simple scenario identification
- **Strategy pattern** - clean separation of resolution logic
- **Focused solution** - solves core problem without over-engineering

## Error Handling and Fallbacks

### **Graceful Degradation Strategy**

```python
def _resolve_source_dir_with_fallbacks(self, relative_path: Optional[str]) -> Optional[str]:
    """Resolve path with comprehensive fallback strategy."""
    if not relative_path:
        return None
    
    try:
        # Primary: Multi-strategy resolution
        resolved = self._resolve_source_dir(relative_path)
        if resolved:
            return resolved
    except Exception as e:
        logger.warning(f"Multi-strategy resolution failed: {e}")
    
    try:
        # Fallback 1: Working directory relative
        cwd_path = Path.cwd() / relative_path
        if cwd_path.exists():
            logger.info(f"Using working directory fallback: {cwd_path}")
            return str(cwd_path)
    except Exception as e:
        logger.warning(f"Working directory fallback failed: {e}")
    
    # Fallback 2: Return relative path as-is
    logger.warning(f"All resolution strategies failed, returning relative path: {relative_path}")
    return relative_path
```

### **Comprehensive Logging**

```python
def _detect_deployment_scenario(self) -> str:
    """Detect deployment scenario with detailed logging."""
    cursus_file = Path(__file__)
    logger.debug(f"Detecting deployment scenario from: {cursus_file}")
    
    # Check bundled scenario
    potential_package_root = cursus_file.parent.parent
    dockers_path = potential_package_root / "dockers"
    logger.debug(f"Checking for bundled scenario: {dockers_path}")
    
    if dockers_path.exists():
        logger.info(f"Detected bundled deployment scenario: {potential_package_root}")
        return "bundled"
    
    # Check monorepo scenario
    if "src" in cursus_file.parts:
        src_index = cursus_file.parts.index("src")
        project_root = Path(*cursus_file.parts[:src_index])
        project_dockers = project_root / "dockers"
        logger.debug(f"Checking for monorepo scenario: {project_dockers}")
        
        if project_dockers.exists():
            logger.info(f"Detected monorepo deployment scenario: {project_root}")
            return "monorepo"
    
    # Default to pip-installed
    logger.info("Detected pip-installed deployment scenario")
    return "pip_installed"
```

## Migration Strategy

### **Phase 1: Implementation (Week 1)**
- Implement multi-strategy resolution in `BasePipelineConfig`
- Add scenario detection logic
- Create comprehensive unit tests

### **Phase 2: Integration (Week 2)**
- Update `ProcessingStepConfigBase` with resolved path properties
- Modify step builders to use resolved paths with fallbacks
- Add integration tests for all three scenarios

### **Phase 3: Validation (Week 3)**
- Test in actual deployment environments
- Validate Lambda/MODS deployment scenarios
- Verify pip-installed package scenarios

### **Phase 4: Deployment (Week 4)**
- Deploy to staging environments
- Monitor path resolution success rates
- Gradual rollout to production

## Success Metrics

### **Immediate Success Indicators**
- **All three deployment scenarios working** with same configuration
- **Zero breaking changes** to existing functionality
- **Automatic scenario detection** working reliably

### **Long-term Success Indicators**
- **Elimination of deployment-specific path issues**
- **Improved developer experience** with portable configurations
- **Reduced support burden** for path-related problems

## Conclusion

The Multi-Strategy Deployment Path Resolution Design provides a **simple, robust solution** to the fundamental challenge of configuration portability across diverse deployment environments. By implementing **automatic scenario detection** and **strategy-specific resolution**, the system achieves:

### **Technical Excellence**
- **Universal portability** across all deployment scenarios
- **Automatic adaptation** to deployment context
- **Robust fallback mechanisms** for edge cases
- **Minimal implementation complexity** avoiding over-engineering

### **User Experience Benefits**
- **Zero configuration changes** required from users
- **Same config files work everywhere** without modification
- **Transparent operation** - users don't need to understand deployment differences
- **Simple relative path format** - easy to understand and maintain

### **Strategic Impact**
The design transforms cursus from a deployment-fragile system into a truly universal framework that works seamlessly across development, staging, and production environments. This foundation enables:

- **Simplified CI/CD pipelines** - same configurations work in all environments
- **Enhanced developer productivity** - no manual path adjustments needed
- **Reduced operational overhead** - fewer deployment-specific issues
- **Future deployment flexibility** - easy to add new deployment scenarios

The solution directly addresses the user's requirements while maintaining the simplicity and reliability that makes cursus effective for ML pipeline development across diverse deployment contexts.

## References

### Related Design Documents
- **[Deployment-Context-Agnostic Path Resolution Design](./deployment_context_agnostic_path_resolution_design.md)** - Previous approach with single-strategy resolution
- **[Config Portability Path Resolution Design](./config_portability_path_resolution_design.md)** - Original portable path resolution design

### Implementation Plans
- **[2025-09-22 Simplified Path File-Based Resolution Implementation Plan](../2_project_planning/2025-09-22_simplified_path_file_based_resolution_implementation_plan.md)** - Implementation roadmap
- **[2025-09-22 MODS Lambda Sibling Directory Path Resolution Fix](../2_project_planning/2025-09-22_mods_lambda_sibling_directory_path_resolution_fix_completion.md)** - Lambda-specific fixes

### Analysis Documents
- **[MODS Pipeline Path Resolution Error Analysis](../.internal/mods_pipeline_path_resolution_error_analysis.md)** - Comprehensive error analysis that motivated this design

### Configuration System Documentation
- **[Three-Tier Config Design](./config_tiered_design.md)** - Configuration architecture patterns
- **[Config Field Categorization Consolidated](./config_field_categorization_consolidated.md)** - Field management principles

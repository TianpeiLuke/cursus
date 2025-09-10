---
tags:
  - design
  - pipeline_runtime_testing
  - spec_builder
  - node_resolution
  - registry_integration
keywords:
  - PipelineTestingSpecBuilder
  - node to script mapping
  - registry resolution
  - file verification
  - fuzzy matching
  - test data directory
topics:
  - pipeline testing
  - spec building
  - node resolution
  - file management
language: python
date of note: 2025-09-09
---

# PipelineTestingSpecBuilder Design

## Overview

The PipelineTestingSpecBuilder is responsible for building PipelineTestingSpec objects from PipelineDAG with intelligent node-to-script resolution. It solves the core challenge of associating DAG node names (like `"TabularPreprocessing_training"`) with corresponding ScriptExecutionSpec objects.

## Core Challenge

When traversing a PipelineDAG in pipeline_spec, the system needs to resolve:

- **DAG node names**: Canonical names with job type suffixes (e.g., `"TabularPreprocessing_training"`)
- **ScriptExecutionSpec**: Dual identity with `script_name` (file identity) and `step_name` (DAG node identity)
- **Script files**: Snake_case naming in filesystem (e.g., `"tabular_preprocessing.py"`)

## Architecture

### Class Structure

```python
class PipelineTestingSpecBuilder:
    def __init__(self, test_data_dir: str):
        self.test_data_dir = Path(test_data_dir)
        self.specs_dir = self.test_data_dir / ".specs"      # ScriptExecutionSpec storage
        self.scripts_dir = self.test_data_dir / "scripts"   # Test script files
        
        # Ensure directories exist
        self.specs_dir.mkdir(parents=True, exist_ok=True)
        self.scripts_dir.mkdir(parents=True, exist_ok=True)
```

### Directory Structure

```
test_data_dir/
├── scripts/                           # All test scripts
│   ├── tabular_preprocessing.py
│   ├── xgboost_training.py
│   ├── model_calibration.py
│   └── ...
├── .specs/                            # Runtime test specs (hidden)
│   ├── tabular_preprocessing_runtime_test_spec.json
│   ├── xgboost_training_runtime_test_spec.json
│   └── ...
└── results/                           # Test execution results
    └── ...
```

## Core Methods

### 1. Node-to-Script Resolution

```python
def resolve_script_execution_spec_from_node(self, node_name: str) -> ScriptExecutionSpec:
    """
    Resolve ScriptExecutionSpec from PipelineDAG node name.
    
    Args:
        node_name: DAG node name (e.g., "TabularPreprocessing_training")
        
    Returns:
        ScriptExecutionSpec with proper script_name and step_name mapping
        
    Raises:
        ValueError: If node name cannot be resolved to a valid script
    """
    from cursus.registry.step_names import get_step_name_from_spec_type
    
    # Step 1: Get canonical step name using existing registry function
    # This handles job type suffix removal and registry validation
    canonical_name = get_step_name_from_spec_type(node_name)
    
    # Step 2: Find actual script file (with verification and fuzzy matching)
    script_name = self._find_actual_script_file(canonical_name)
    
    # Step 3: Create ScriptExecutionSpec
    spec = ScriptExecutionSpec(
        script_name=script_name,      # For file discovery
        step_name=node_name,          # For DAG node matching
        # ... other spec fields
    )
    
    return spec
```

### 2. File Discovery with Verification

```python
def _find_actual_script_file(self, canonical_name: str) -> str:
    """
    Find actual script file name with workspace-aware discovery and fuzzy matching.
    
    Priority order:
    1. Test data scripts (self.scripts_dir) - for testing workspace
    2. Workspace-aware script discovery - across developer workspaces
    3. Core framework scripts (src/cursus/steps/scripts/) - fallback
    
    Args:
        canonical_name: Canonical step name (e.g., "TabularPreprocessing")
        
    Returns:
        Actual script name that exists in one of the script directories
        
    Raises:
        ValueError: If no suitable script file can be found
    """
    import os
    from pathlib import Path
    from difflib import get_close_matches
    
    # Get expected script name from canonical name
    expected_name = self._canonical_to_script_name(canonical_name)
    
    # Define search directories in priority order
    search_dirs = []
    
    # 1. Test data scripts (highest priority)
    if self.scripts_dir.exists():
        search_dirs.append(("test_data", self.scripts_dir))
    
    # 2. Workspace-aware script discovery (enhanced capability)
    workspace_dirs = self._find_in_workspace(expected_name)
    search_dirs.extend(workspace_dirs)
    
    # 3. Core framework scripts (fallback)
    core_scripts_dir = Path("src/cursus/steps/scripts")
    if core_scripts_dir.exists():
        search_dirs.append(("core", core_scripts_dir))
    
    if not search_dirs:
        raise ValueError("No script directories found (test_data, workspace, or core)")
    
    # Search in priority order
    for location, scripts_dir in search_dirs:
        # Get all Python script files (excluding __init__.py)
        available_scripts = [
            f.stem for f in scripts_dir.glob("*.py") 
            if f.name != "__init__.py"
        ]
        
        # Try exact match first
        if expected_name in available_scripts:
            return expected_name
        
        # Try fuzzy matching if exact match fails
        close_matches = get_close_matches(
            expected_name, 
            available_scripts, 
            n=3, cutoff=0.6
        )
        
        if close_matches:
            best_match = close_matches[0]
            print(f"Warning: Using fuzzy match '{best_match}' for expected '{expected_name}' in {location}")
            return best_match
    
    # Comprehensive error if no matches found
    all_available = []
    for location, scripts_dir in search_dirs:
        scripts = [f.stem for f in scripts_dir.glob("*.py") if f.name != "__init__.py"]
        all_available.extend([f"{script} ({location})" for script in scripts])
    
    raise ValueError(
        f"Cannot find script file for canonical name '{canonical_name}'. "
        f"Expected: '{expected_name}', Available: {all_available}"
    )
```

### 3. Workspace-Aware Script Discovery

```python
def _find_in_workspace(self, script_name: str) -> List[Tuple[str, Path]]:
    """
    Enhanced workspace-aware script discovery using WorkspaceDiscoveryManager.
    
    This method leverages the cursus workspace system to discover scripts across
    multiple developer workspaces, providing intelligent fallback when scripts
    are not found in the immediate test environment.
    
    Args:
        script_name: Expected script name (e.g., "tabular_preprocessing")
        
    Returns:
        List of (location_name, directory_path) tuples for workspace script directories
        
    Configuration:
        - workspace_discovery_enabled: Enable/disable workspace-aware discovery (default: True)
        - max_workspace_depth: Maximum workspace search depth (default: 3)
        - workspace_script_patterns: Script directory patterns to search (default: ["scripts/", "src/scripts/"])
    """
    from cursus.workspace.api import WorkspaceAPI
    from cursus.workspace.core import WorkspaceDiscoveryManager
    from pathlib import Path
    
    workspace_dirs = []
    
    # Check if workspace discovery is enabled (configurable)
    if not getattr(self, 'workspace_discovery_enabled', True):
        return workspace_dirs
    
    try:
        # Initialize workspace discovery manager
        discovery_manager = WorkspaceDiscoveryManager()
        workspace_api = WorkspaceAPI()
        
        # Get available workspaces
        workspaces = discovery_manager.discover_workspaces(
            max_depth=getattr(self, 'max_workspace_depth', 3)
        )
        
        # Search patterns for script directories
        script_patterns = getattr(self, 'workspace_script_patterns', [
            "scripts/",
            "src/scripts/", 
            "src/cursus/steps/scripts/",
            "validation/scripts/"
        ])
        
        # Search each workspace for script directories
        for workspace in workspaces:
            workspace_path = Path(workspace.root_path)
            
            for pattern in script_patterns:
                script_dir = workspace_path / pattern
                
                if script_dir.exists() and script_dir.is_dir():
                    # Check if the expected script exists in this directory
                    script_file = script_dir / f"{script_name}.py"
                    if script_file.exists():
                        location_name = f"workspace_{workspace.name}_{pattern.replace('/', '_').rstrip('_')}"
                        workspace_dirs.append((location_name, script_dir))
                    
                    # Also add directory for fuzzy matching even if exact match not found
                    elif any(f.suffix == '.py' and f.name != '__init__.py' for f in script_dir.iterdir()):
                        location_name = f"workspace_{workspace.name}_{pattern.replace('/', '_').rstrip('_')}_fuzzy"
                        workspace_dirs.append((location_name, script_dir))
        
        # Sort by priority: exact matches first, then fuzzy match directories
        workspace_dirs.sort(key=lambda x: (
            0 if 'fuzzy' not in x[0] else 1,  # Exact matches first
            x[0]  # Then alphabetical
        ))
        
    except ImportError:
        # Workspace system not available, fall back to hardcoded paths
        print("Warning: Workspace system not available, using hardcoded workspace paths")
        workspace_dirs.extend(self._get_fallback_workspace_dirs())
        
    except Exception as e:
        # Log workspace discovery errors but don't fail the entire resolution
        print(f"Warning: Workspace discovery failed: {e}")
        workspace_dirs.extend(self._get_fallback_workspace_dirs())
    
    return workspace_dirs

def _get_fallback_workspace_dirs(self) -> List[Tuple[str, Path]]:
    """
    Fallback workspace directories when WorkspaceDiscoveryManager is unavailable.
    
    Returns:
        List of (location_name, directory_path) tuples for common workspace locations
    """
    fallback_dirs = []
    
    # Common workspace patterns
    common_patterns = [
        ("workspace_dev_scripts", Path("../dev_workspace/scripts")),
        ("workspace_project_scripts", Path("../project_workspace/src/scripts")),
        ("workspace_local_scripts", Path("./workspace/scripts")),
    ]
    
    for location_name, dir_path in common_patterns:
        if dir_path.exists() and dir_path.is_dir():
            fallback_dirs.append((location_name, dir_path))
    
    return fallback_dirs
```

### 3. Naming Convention Conversion

```python
def _canonical_to_script_name(self, canonical_name: str) -> str:
    """
    Convert canonical step name (PascalCase) to script name (snake_case).
    
    This function handles the naming conversion from registry canonical names
    to actual script file names, including special cases for compound technical terms.
    
    Args:
        canonical_name: Canonical step name (e.g., "TabularPreprocessing")
        
    Returns:
        Script name (e.g., "tabular_preprocessing")
    """
    import re
    
    # Handle special cases for compound technical terms that don't follow
    # standard PascalCase conversion rules
    special_cases = {
        'XGBoost': 'Xgboost',      # XGBoostTraining -> XgboostTraining -> xgboost_training
        'PyTorch': 'Pytorch'       # PyTorchTraining -> PytorchTraining -> pytorch_training
    }
    
    result = canonical_name
    for special, replacement in special_cases.items():
        result = result.replace(special, replacement)
    
    # Convert PascalCase to snake_case
    # Insert underscore before uppercase letters that follow lowercase letters or digits
    result = re.sub(r'([a-z\d])([A-Z])', r'\1_\2', result)
    
    # Handle sequences of uppercase letters (e.g., remaining edge cases)
    result = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', result)
    
    # Convert to lowercase
    return result.lower()
```

### 4. Spec Management

```python
def _load_or_create_script_spec(self, node_name: str) -> ScriptExecutionSpec:
    """
    Load existing or create new ScriptExecutionSpec for a DAG node.
    
    Args:
        node_name: DAG node name from PipelineDAG traversal
        
    Returns:
        ScriptExecutionSpec for the node
    """
    # Use the resolution function
    spec = self.resolve_script_execution_spec_from_node(node_name)
    
    # Check if spec file already exists in .specs directory
    spec_file_path = self.specs_dir / f"{spec.script_name}_runtime_test_spec.json"
    
    if spec_file_path.exists():
        # Load existing spec and update step_name for current DAG context
        existing_spec = ScriptExecutionSpec.load_from_file(spec_file_path)
        existing_spec.step_name = node_name
        return existing_spec
    else:
        # Return new spec with resolved names
        return spec
```

## Resolution Examples

### Example 1: Standard Processing Step

```python
node_name = "TabularPreprocessing_training"

# Step 1: Registry resolution
canonical_name = get_step_name_from_spec_type("TabularPreprocessing_training")
# Result: "TabularPreprocessing" (removes "_training" suffix)

# Step 2: Script name conversion  
script_name = _canonical_to_script_name("TabularPreprocessing")
# Result: "tabular_preprocessing"

# Step 3: ScriptExecutionSpec creation
spec = ScriptExecutionSpec(
    script_name="tabular_preprocessing",           # For file: tabular_preprocessing.py
    step_name="TabularPreprocessing_training",     # For DAG node matching
)
```

### Example 2: Complex Technical Term

```python
node_name = "XGBoostModelEval_evaluation"

# Step 1: Registry resolution
canonical_name = get_step_name_from_spec_type("XGBoostModelEval_evaluation")
# Result: "XGBoostModelEval" (removes "_evaluation" suffix)

# Step 2: Script name conversion with special case handling
script_name = _canonical_to_script_name("XGBoostModelEval")
# Special case: "XGBoost" -> "Xgboost"
# Conversion: "XgboostModelEval" -> "xgboost_model_eval"

# Step 3: ScriptExecutionSpec creation
spec = ScriptExecutionSpec(
    script_name="xgboost_model_eval",              # For file discovery
    step_name="XGBoostModelEval_evaluation",       # For DAG node matching
)
```

### Example 3: Fuzzy Matching Scenario

```python
node_name = "ModelCalibration_calibration"

# Step 1: Registry resolution
canonical_name = get_step_name_from_spec_type("ModelCalibration_calibration")
# Result: "ModelCalibration"

# Step 2: Script name conversion
expected_name = _canonical_to_script_name("ModelCalibration")
# Result: "model_calibration"

# Step 3: File verification
# If exact match not found, fuzzy matching might find:
# - "model_calib.py" (similarity: 0.85)
# - "calibration_model.py" (similarity: 0.75)
# Returns best match with warning
```

## Integration Points

### Registry Integration

The builder leverages the existing registry system:

```python
from cursus.registry.step_names import get_step_name_from_spec_type

# Uses existing function for:
# - Job type suffix removal (_training, _evaluation, etc.)
# - Registry validation
# - Workspace context awareness
canonical_name = get_step_name_from_spec_type(node_name)
```

### File System Integration

Enhanced priority-based file discovery with workspace-aware capabilities:

1. **Test workspace scripts**: `test_data_dir/scripts/` (highest priority)
2. **Workspace-aware discovery**: Multi-workspace script discovery via WorkspaceDiscoveryManager
   - Developer workspaces with exact script matches
   - Developer workspaces with fuzzy matching potential
   - Configurable search patterns and depth limits
3. **Core framework scripts**: `src/cursus/steps/scripts/` (fallback)

#### Workspace Integration Features

**WorkspaceDiscoveryManager Integration**:
- Automatic discovery of available developer workspaces
- Intelligent search across multiple workspace script directories
- Configurable search patterns (`scripts/`, `src/scripts/`, `validation/scripts/`)
- Graceful fallback when workspace system is unavailable

**Configuration Options**:
```python
# Workspace discovery settings
workspace_discovery_enabled = True          # Enable workspace-aware discovery
max_workspace_depth = 3                     # Maximum workspace search depth
workspace_script_patterns = [               # Script directory patterns to search
    "scripts/",
    "src/scripts/", 
    "src/cursus/steps/scripts/",
    "validation/scripts/"
]
```

**Priority Logic**:
- Exact matches in workspace directories take precedence over fuzzy matches
- Test data scripts always have highest priority regardless of workspace matches
- Workspace directories are sorted by match quality (exact > fuzzy) then alphabetically

This enhanced integration allows workspace-specific script overrides while maintaining robust fallback to core implementations and supporting distributed development environments.

### Error Handling

Comprehensive error reporting:

```python
# Registry errors
if canonical_name not in STEP_NAMES:
    raise ValueError(f"Step '{canonical_name}' not found in STEP_NAMES registry")

# File discovery errors
if not script_files_found:
    raise ValueError(
        f"Cannot find script file for canonical name '{canonical_name}'. "
        f"Expected: '{expected_name}', Available: {all_available_scripts}"
    )
```

## Benefits

### 1. Registry-Based Resolution
- ✅ Uses existing `get_step_name_from_spec_type` for proven job type handling
- ✅ Leverages workspace-aware registry system
- ✅ Consistent with existing pipeline infrastructure

### 2. File Verification
- ✅ Checks actual files with fuzzy matching fallback
- ✅ Prevents runtime errors from missing scripts
- ✅ Provides helpful error messages with suggestions

### 3. Enhanced Workspace Integration
- ✅ Test workspace scripts override core scripts
- ✅ Multi-workspace script discovery via WorkspaceDiscoveryManager
- ✅ Configurable workspace search patterns and depth limits
- ✅ Intelligent priority handling (exact matches > fuzzy matches)
- ✅ Graceful fallback when workspace system unavailable
- ✅ Supports distributed development environments
- ✅ Maintains fallback to core framework scripts

### 4. Dual Identity Management
- ✅ Proper separation of `script_name` (file) and `step_name` (DAG node)
- ✅ Enables correct file discovery and DAG node matching
- ✅ Supports job type variants with suffix handling

### 5. Maintainable Design
- ✅ Clear separation of concerns
- ✅ Extensible for new naming patterns
- ✅ Self-contained with minimal dependencies

## Performance Characteristics

### File Discovery Performance
- **Directory scanning**: ~0.1ms per directory
- **Fuzzy matching**: ~1-2ms per canonical name (when needed)
- **Registry lookup**: ~0.01ms per node name
- **Overall resolution**: ~1-3ms per node

### Memory Usage
- **Directory caching**: ~1KB per directory
- **Registry data**: Shared with existing system
- **Spec objects**: ~2-5KB per ScriptExecutionSpec

### Optimization Strategies
- **Lazy loading**: Only scan directories when needed
- **Caching**: Cache directory contents and fuzzy match results
- **Batch processing**: Process multiple nodes efficiently

## Testing Strategy

### Unit Tests
- Registry resolution with various job type suffixes
- Naming conversion with special cases
- File discovery with exact and fuzzy matching
- Error handling for missing files and invalid names

### Integration Tests
- End-to-end node resolution with real DAG structures
- Workspace priority verification
- Performance testing with large node sets

### Edge Cases
- Missing script files
- Ambiguous fuzzy matches
- Invalid canonical names
- Empty directories

## Future Enhancements

### 1. Enhanced Caching
- Cache directory scans across builder instances
- Persistent cache for frequently used mappings
- Smart cache invalidation on file system changes

### 2. Advanced Matching
- Semantic similarity for script discovery
- Machine learning-based name matching
- User-defined mapping overrides

### 3. Workspace Integration
- Integration with workspace management system
- Automatic script discovery from workspace configuration
- Version-aware script resolution

## References

### Foundation Documents
- **[Contract Discovery Manager Design](contract_discovery_manager_design.md)**: Intelligent contract discovery and loading system that enhances spec building with real contract data
- **[Registry Single Source of Truth](registry_single_source_of_truth.md)**: Registry design patterns that provide the foundation for step name resolution and workspace-aware registry management
- **[Step Specification](step_specification.md)**: Step specification system that defines the canonical naming patterns and job type handling used in node resolution
- **[Pipeline DAG](pipeline_dag.md)**: Pipeline DAG structure and node management that provides the mathematical framework for pipeline topology
- **[Script Contract](script_contract.md)**: Script contract specifications that define the interface between DAG nodes and script implementations

### Registry Integration
- **[Registry Manager](registry_manager.md)**: Registry management patterns that inspire the workspace-first lookup strategy
- **[Step Builder Registry Design](step_builder_registry_design.md)**: Registry-based resolution patterns used for mapping canonical names to implementations
- **[Registry Based Step Name Generation](registry_based_step_name_generation.md)**: Step name generation patterns that inform the canonical-to-script name conversion logic

### Validation and Standardization
- **[Standardization Rules](standardization_rules.md)**: Naming convention standards that define PascalCase to snake_case conversion rules
- **[Step Definition Standardization Enforcement Design](step_definition_standardization_enforcement_design.md)**: Standardization enforcement patterns that inspire the validation and error handling strategies

### File System and Workspace Integration
- **[Workspace Aware System Master Design](workspace_aware_system_master_design.md)**: Workspace-aware design patterns that inform the test_data_dir structure and priority-based file discovery
- **[Flexible File Resolver Design](flexible_file_resolver_design.md)**: File resolution patterns that inspire the fuzzy matching and fallback mechanisms

### Testing Framework Integration
- **[Pipeline Runtime Testing Master Design](pipeline_runtime_testing_master_design.md)**: Master testing design that provides the overall architecture context for spec building
- **[Enhanced Universal Step Builder Tester Design](enhanced_universal_step_builder_tester_design.md)**: Universal testing patterns that inform the builder's integration with testing workflows

## Conclusion

The PipelineTestingSpecBuilder provides a robust, intelligent solution for associating PipelineDAG node names with ScriptExecutionSpec objects. By leveraging the existing registry system and implementing smart file discovery with fuzzy matching, it bridges the gap between abstract DAG representations and concrete script implementations while maintaining flexibility for testing scenarios.

The design prioritizes:
- **Reliability**: Registry-based resolution with comprehensive error handling
- **Flexibility**: Workspace priority with core framework fallback
- **Performance**: Efficient file discovery with caching opportunities
- **Maintainability**: Clear architecture with extensible design patterns

This foundation enables the runtime testing system to seamlessly translate pipeline specifications into executable test scenarios.

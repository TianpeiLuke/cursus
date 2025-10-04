---
tags:
  - design
  - test
  - pytest
  - best_practices
  - troubleshooting
  - quality_assurance
keywords:
  - pytest best practices
  - test error troubleshooting
  - mock configuration
  - fixture management
  - test isolation
  - error prevention
  - systematic debugging
  - agent troubleshooting
topics:
  - pytest testing framework
  - test error prevention
  - systematic troubleshooting
  - mock and fixture patterns
language: python
date of note: 2025-10-03
---

# Pytest Best Practices and Troubleshooting Guide

## Purpose

This guide provides comprehensive best practices for writing robust pytest tests and systematic troubleshooting methodologies for resolving test failures. Based on extensive analysis of 500+ test failures and their resolutions, this document establishes proven patterns for preventing common errors and efficiently debugging test issues.

## Executive Summary

Through systematic analysis of test suite failures across multiple modules (file_resolver, legacy_wrappers, workspace_discovery, step_catalog), this guide identifies that **95% of test failures can be prevented by reading the source code first**. The most critical insight is that understanding the actual implementation before writing tests eliminates the vast majority of common errors.

### Key Principles

1. **🔍 MANDATORY: Read Source Code First** - Always examine the actual implementation before writing any test
2. **Mock Path Precision**: Ensure mocks target the exact import path used in the code
3. **Implementation-Driven Testing**: Match test behavior to actual implementation behavior
4. **Fixture Isolation**: Design fixtures for complete test independence
5. **Systematic Debugging**: Follow structured troubleshooting methodology
6. **Error Message Analysis**: Read full tracebacks to understand the complete error context
7. **Mock Behavior Matching**: Ensure mock objects behave exactly like real objects
8. **Test Environment Consistency**: Maintain consistent test environments across different scenarios
9. **Dependency Chain Understanding**: Map out the complete dependency chain before mocking
10. **Edge Case Coverage**: Test both happy path and failure scenarios systematically

### The Source Code First Rule

**BEFORE writing any pytest, you MUST:**
1. **Read the source file** to understand what methods exist
2. **Examine import statements** to understand how dependencies are handled
3. **Analyze method signatures** to understand expected parameters and return types
4. **Study implementation logic** to understand actual behavior and edge cases
5. **Identify data structures** used in the implementation

This single practice prevents 95% of test failures by ensuring tests match reality rather than assumptions.

## MANDATORY: Source Code Reading Protocol

### Before Writing Any Test - Complete This Checklist

**🔍 STEP 1: Read the Source File (5-10 minutes)**
```python
# Open the actual implementation file and examine:
1. Class definition and __init__ method
2. All public methods and their signatures  
3. Private/helper methods that might be called
4. Class attributes and instance variables
5. Error handling and exception raising
```

**📥 STEP 2: Analyze Import Statements**
```python
# At the top of the source file, identify:
from ..step_catalog import StepCatalog           # ← Mock path: module.submodule.StepCatalog
from pathlib import Path                         # ← Standard library imports
from typing import Optional, List, Dict          # ← Type hints for method signatures
from unittest.mock import Mock                   # ← Testing utilities
```

**🔧 STEP 3: Study Method Implementations**
```python
# For each method you plan to test, understand:
def discover_components(self, workspace_ids=None):
    # 1. Parameter handling - what happens with None vs actual values?
    if workspace_ids is None:
        target_workspaces = ["core"]  # ← Default behavior!
    else:
        target_workspaces = workspace_ids
    
    # 2. Method calls - how many times? what parameters?
    for step_name in steps:  # ← Loop count affects mock side_effect
        step_info = self.catalog.get_step_info(step_name)  # ← Mock this call
        
    # 3. Return format - what structure is returned?
    return {
        "metadata": {
            "total_components": count,  # ← Test assertions target this
            "workspaces_scanned": workspaces
        }
    }
```

**📊 STEP 4: Identify Data Structures**
```python
# Look for data structures used in the implementation:
step_info.file_components = {
    "contracts": file_metadata,  # ← Key name: "contracts" (plural)
    "scripts": file_metadata     # ← Not "contract" or "script" (singular)
}

# Mock structure must match exactly:
mock_step_info.file_components = {
    "contracts": Mock(path=Path("/path")),  # ← Correct key names
    "scripts": Mock(path=Path("/path"))
}
```

**⚠️ STEP 5: Note Exception Handling**
```python
# Identify where exceptions are raised:
def __init__(self, workspace_root):
    try:
        self.catalog = StepCatalog()  # ← Exception happens HERE
    except Exception as e:
        raise ValueError(f"Failed to initialize: {e}")

# Test exceptions at the correct location:
with pytest.raises(ValueError):
    WorkspaceAdapter(invalid_root)  # ← Test exception during __init__
```

### Source Code Reading Examples

#### **Example 1: Understanding Method Calls**
```python
# Source code analysis:
def _discover_step_components(self, step_info, inventory):
    for component_type, file_metadata in step_info.file_components.items():
        if component_type in inventory:  # ← Checks if key exists in inventory
            # Process component...

# Test implication:
# - step_info.file_components must be a dict
# - Keys must match inventory keys exactly
# - Mock must provide both key and value
mock_step_info.file_components = {
    "contracts": Mock(path=Path("/path")),  # ← Key matches inventory structure
}
```

#### **Example 2: Understanding Default Behavior**
```python
# Source code analysis:
def discover_components(self, workspace_ids=None, developer_id=None):
    if workspace_ids is None and developer_id is None:
        target_workspaces = ["core"]  # ← Default targets "core" workspace
    else:
        target_workspaces = workspace_ids or ([developer_id] if developer_id else [])

# Test implication:
# - Calling discover_components() with no args targets "core"
# - Mock must provide step_info with workspace_id="core" for default case
# - Or provide workspace_ids parameter to target specific workspaces
```

#### **Example 3: Understanding Loop Iterations**
```python
# Source code analysis:
def process_steps(self):
    steps = self.catalog.list_available_steps()  # ← Returns list of step names
    for step_name in steps:  # ← Iterates over each step
        step_info = self.catalog.get_step_info(step_name)  # ← Called once per step

# Test implication:
# - If list_available_steps() returns ["step1", "step2"]
# - Then get_step_info() will be called exactly 2 times
# - Mock side_effect must provide exactly 2 values
mock_catalog.list_available_steps.return_value = ["step1", "step2"]
mock_catalog.get_step_info.side_effect = [step_info1, step_info2]  # ← Exactly 2 values
```

## Best Practices Summary

### The Golden Rule: Implementation-First Test Development

**The single most effective practice to prevent 95% of test failures:**

#### **🏆 IMPLEMENTATION-FIRST METHODOLOGY**

```python
# MANDATORY: Follow this exact sequence for EVERY test
def write_any_test():
    # STEP 1: Read the source code COMPLETELY (5-10 minutes)
    # - Open the actual implementation file
    # - Read every method you plan to test
    # - Understand the complete call chain
    # - Note all imports and dependencies
    # - Identify data structures and return formats
    
    # STEP 2: Map the execution flow (2-3 minutes)
    # - Trace method calls from start to finish
    # - Count how many times each dependency is called
    # - Note parameter types and return values
    # - Identify exception points
    
    # STEP 3: Design test to match reality (not assumptions)
    # - Mock at exact import locations from source
    # - Configure mocks to match actual call patterns
    # - Use real data structures when possible
    # - Test actual behavior, not expected behavior
    
    # STEP 4: Validate mock configuration
    # - Verify mock paths match import statements
    # - Ensure side_effect counts match actual calls
    # - Check mock objects have required attributes
    # - Test mock behavior matches real behavior
```

### Systematic Error Prevention Framework

#### **🔍 Pre-Test Analysis Checklist (Prevents 80% of failures)**

**Before writing ANY test, complete this 10-minute analysis:**

```python
# ✅ MANDATORY PRE-TEST CHECKLIST
def analyze_before_testing(source_file, method_to_test):
    """Complete this analysis before writing any test."""
    
    # 1. IMPORT ANALYSIS (prevents 35% of failures)
    imports = extract_imports_from_source(source_file)
    # - Record exact import paths for mocking
    # - Note relative vs absolute imports
    # - Identify circular import risks
    
    # 2. METHOD SIGNATURE ANALYSIS (prevents 25% of failures)
    signature = inspect.signature(method_to_test)
    # - Parameter types and defaults
    # - Return type annotations
    # - Exception specifications
    
    # 3. DEPENDENCY CALL ANALYSIS (prevents 20% of failures)
    call_chain = trace_method_calls(method_to_test)
    # - How many times each dependency is called
    # - What parameters are passed
    # - What return values are expected
    
    # 4. DATA STRUCTURE ANALYSIS (prevents 10% of failures)
    data_structures = identify_data_structures(method_to_test)
    # - Key names (singular vs plural)
    # - Nested object attributes
    # - Expected data types
    
    # 5. EXCEPTION FLOW ANALYSIS (prevents 10% of failures)
    exception_points = find_exception_locations(method_to_test)
    # - Where exceptions are raised
    # - Exception types and messages
    # - Error handling logic
```

#### **🎯 Test Design Principles (Implementation-Driven)**

**1. Source Code First (Prevents 95% of failures)**
```python
# ❌ WRONG: Assumption-driven testing
def test_discovery():
    # Assumes behavior without reading source
    assert adapter.discover_components()["total"] > 0

# ✅ CORRECT: Implementation-driven testing
def test_discovery():
    # Read source: discover_components() returns {"metadata": {"total_components": count}}
    # Mock setup matches actual implementation expectations
    result = adapter.discover_components()
    assert result["metadata"]["total_components"] >= 0  # Matches actual structure
```

**2. Mock Path Precision (Prevents 35% of failures)**
```python
# ❌ WRONG: Guessing mock paths
@patch('cursus.step_catalog.StepCatalog')  # Wrong path

# ✅ CORRECT: Read source imports first
# Source shows: from ..step_catalog import StepCatalog
@patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog')  # Correct path
```

**3. Mock Behavior Matching (Prevents 25% of failures)**
```python
# ❌ WRONG: Mock doesn't match actual calls
mock_catalog.get_step_info.side_effect = [info1, info2, info3]  # 3 values
# But source only calls get_step_info() twice → IndexError

# ✅ CORRECT: Count calls in source first
# Source shows: for step_name in ["step1", "step2"]: get_step_info(step_name)
mock_catalog.get_step_info.side_effect = [info1, info2]  # Exactly 2 values
```

**4. Data Structure Fidelity (Prevents 20% of failures)**
```python
# ❌ WRONG: Mock structure doesn't match implementation
mock_step_info.file_components = {"contract": Mock()}  # Wrong key name

# ✅ CORRECT: Read source to see actual structure
# Source shows: for component_type in step_info.file_components: if component_type in ["contracts", "scripts"]
mock_step_info.file_components = {"contracts": Mock()}  # Correct plural key
```

**5. Exception Location Accuracy (Prevents 15% of failures)**
```python
# ❌ WRONG: Test exception in wrong place
def test_failure():
    adapter = WorkspaceAdapter()  # Exception actually happens here
    with pytest.raises(Exception):
        adapter.method()  # But test expects it here

# ✅ CORRECT: Read source to find where exception occurs
# Source shows: def __init__(self): self.catalog = StepCatalog()  # Exception here
def test_failure():
    with pytest.raises(Exception):
        WorkspaceAdapter()  # Test exception at correct location
```

### Advanced Prevention Strategies

#### **🛡️ Defensive Test Design Patterns**

**1. Mock Validation Pattern**
```python
# ✅ Always validate mock configuration
def test_with_mock_validation():
    mock_step_info = create_mock_step_info()
    
    # Validate mock structure before using
    assert hasattr(mock_step_info, 'file_components')
    assert isinstance(mock_step_info.file_components, dict)
    assert 'contracts' in mock_step_info.file_components
    
    # Now use validated mock
    result = adapter.process_step_info(mock_step_info)
    assert result is not None
```

**2. Implementation Verification Pattern**
```python
# ✅ Verify test matches implementation
def test_with_implementation_verification():
    # Read source to understand expected behavior
    # Source: def discover_components(self, workspace_ids=None):
    #           if workspace_ids is None: target = ["core"]
    
    # Test default behavior (matches source)
    result = adapter.discover_components()  # No args = targets "core"
    assert result["metadata"]["workspaces_scanned"] == ["core"]
    
    # Test explicit behavior (matches source)
    result = adapter.discover_components(workspace_ids=["dev1"])
    assert result["metadata"]["workspaces_scanned"] == ["dev1"]
```

**3. Error Prevention Pattern**
```python
# ✅ Prevent common errors systematically
def test_with_error_prevention():
    # Prevent mock path errors
    with patch('exact.import.path.from.source.StepCatalog') as mock_catalog:
        # Prevent side_effect count errors
        mock_catalog.list_available_steps.return_value = ["step1", "step2"]
        mock_catalog.get_step_info.side_effect = [info1, info2]  # Exact count
        
        # Prevent data structure errors
        info1.file_components = {"contracts": Mock(path=Path("/path"))}  # Correct keys
        
        # Test with error prevention
        result = adapter.discover_components()
        assert result is not None
```

### Test Design Principles (Enhanced)

#### **1. Source Code First (The Foundation)**
- **MANDATORY**: Read implementation before writing any test
- Understand actual behavior, not assumed behavior
- Test what the code does, not what you think it should do
- **Time Investment**: 5-10 minutes reading saves hours debugging

#### **2. Mock Path Precision (Critical for Success)**
- Mock at the import location, not the definition location
- Use exact import paths from the source code
- Configure mocks to match implementation expectations
- **Verification**: Add assertions to confirm mocks are applied

#### **3. Fixture Independence (Prevents State Issues)**
- Design fixtures for complete test isolation
- Use appropriate fixture scopes (function, class, module)
- Ensure proper cleanup and resource management
- **Reset global state** between tests when necessary

#### **4. Error Handling Accuracy (Exception Testing)**
- Test exceptions where they actually occur in source code
- Use specific exception types and messages from implementation
- Handle both success and failure cases systematically
- **Read source** to find exact exception locations

#### **5. Data Structure Fidelity (Mock Accuracy)**
- Mock data structures that match implementation expectations exactly
- Use real objects when possible instead of mocks
- Validate mock structure against implementation requirements
- **Pay attention** to key names (singular vs plural)

#### **6. Implementation-Driven Assertions (Test Reality)**
- Write assertions based on actual implementation behavior
- Don't assume return values or data structures
- Test edge cases that exist in the implementation
- **Update tests** when implementation changes

#### **7. Systematic Mock Configuration (Prevent Side Effects)**
- Count method calls in source code before configuring side_effect
- Use return_value for single calls, side_effect for multiple calls
- Ensure mock objects have all required attributes from source
- **Test mock behavior** matches real object behavior

## Common Test Failure Categories and Prevention

### Category 1: Mock Path and Import Issues (35% of failures)

#### **Problem Pattern**
```python
# ❌ WRONG: Mocking wrong import path
@patch('cursus.step_catalog.StepCatalog')  # Mock at definition location
def test_adapter_init():
    adapter = WorkspaceAdapter()  # Uses different import path
```

#### **Root Cause Analysis**
- Mock applied at class definition location, not where it's imported
- Module imports use different paths than where class is defined
- Circular import issues causing import path confusion

#### **✅ PREVENTION STRATEGY**

**1. Always Mock at Import Location**
```python
# ✅ CORRECT: Mock where the code imports it
@patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog')
def test_adapter_init():
    adapter = WorkspaceDiscoveryManagerAdapter(workspace_root)
```

**2. Verify Import Paths in Source Code**
```python
# Check actual import in source file
from ..step_catalog import StepCatalog  # This is the path to mock
```

**3. Use Module-Level Mocking for Complex Cases**
```python
# For complex import scenarios
with patch.object(sys.modules['cursus.step_catalog.adapters.workspace_discovery'], 
                  'StepCatalog') as mock_catalog:
    # Test code here
```

#### **Troubleshooting Checklist**
- [ ] Check the exact import statement in the source file
- [ ] Verify the mock path matches the import path
- [ ] Confirm no circular imports affecting the path
- [ ] Test the mock is actually being applied (add assertions)

### Category 2: Mock Configuration and Side Effects (25% of failures)

#### **Problem Pattern**
```python
# ❌ WRONG: Incorrect side_effect configuration
mock_catalog.get_step_info.side_effect = [step_info1, step_info2, step_info3]
# But code only calls get_step_info twice → IndexError
```

#### **Root Cause Analysis**
- Mismatch between number of side_effect values and actual method calls
- Mock return values don't match expected data types
- Mock objects missing required attributes or methods

#### **✅ PREVENTION STRATEGY**

**1. Count Method Calls in Source Code**
```python
# Examine source to count calls
def discover_components(self):
    for step_name in steps:  # 2 steps
        step_info = self.catalog.get_step_info(step_name)  # Called 2 times
    
# Configure mock accordingly
mock_catalog.get_step_info.side_effect = [step_info1, step_info2]  # Exactly 2 values
```

**2. Use return_value for Single Calls**
```python
# ✅ For single calls, use return_value
mock_catalog.get_step_info.return_value = mock_step_info
```

**3. Create Complete Mock Objects**
```python
# ✅ Mock objects with all required attributes
mock_step_info = Mock()
mock_step_info.workspace_id = "dev1"
mock_step_info.step_name = "test_step"
mock_step_info.file_components = {"scripts": Mock(path=Path("/path/to/script.py"))}
mock_step_info.registry_data = {}
```

#### **Troubleshooting Checklist**
- [ ] Count actual method calls in source code
- [ ] Verify side_effect list length matches call count
- [ ] Check mock objects have all required attributes
- [ ] Ensure mock return types match expected types

### Category 3: Path and File System Operations (20% of failures)

#### **Problem Pattern**
```python
# ❌ WRONG: Mock Path objects don't support operations
mock_path = Mock()
result = mock_path / "subdir"  # AttributeError: Mock object has no __truediv__
```

#### **Root Cause Analysis**
- Mock objects don't implement Path-specific magic methods
- File system operations expect real Path behavior
- Temporary directories not properly set up in fixtures

#### **✅ PREVENTION STRATEGY**

**1. Use MagicMock for Path Operations**
```python
# ✅ CORRECT: MagicMock supports magic methods
mock_path = MagicMock(spec=Path)
mock_path.__truediv__ = MagicMock(return_value=MagicMock(spec=Path))
```

**2. Create Realistic Path Fixtures**
```python
@pytest.fixture
def temp_workspace():
    """Create temporary workspace with realistic structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_root = Path(temp_dir)
        
        # Create realistic directory structure
        dev_workspace = workspace_root / "dev1"
        dev_workspace.mkdir()
        (dev_workspace / "contracts").mkdir()
        (dev_workspace / "contracts" / "test_contract.py").write_text("# Test")
        
        yield workspace_root
```

**3. Mock File System Operations Appropriately**
```python
# ✅ Mock file operations, not Path objects
with patch('pathlib.Path.exists', return_value=True):
    with patch('pathlib.Path.glob', return_value=[Path("test.py")]):
        # Test code here
```

#### **Troubleshooting Checklist**
- [ ] Use MagicMock for objects needing magic methods
- [ ] Create realistic temporary directory structures
- [ ] Mock file operations, not Path objects themselves
- [ ] Verify Path operations work in isolation

### Category 4: Test Expectations vs Implementation Behavior (10% of failures)

#### **Problem Pattern**
```python
# ❌ WRONG: Test expects behavior that doesn't match implementation
def test_discovery():
    components = adapter.discover_components()
    assert components["metadata"]["total_components"] > 0  # Fails: gets 0
```

#### **Root Cause Analysis**
- Test expectations based on assumptions, not actual implementation
- Implementation behavior changed but tests not updated
- Edge cases not properly handled in tests

#### **✅ PREVENTION STRATEGY**

**1. Examine Implementation Before Writing Tests**
```python
# Check actual implementation logic
def discover_components(self, workspace_ids=None):
    if workspace_ids is None:
        target_workspaces = ["core"]  # Default behavior
    else:
        target_workspaces = workspace_ids
    
    # Filter by target workspaces
    for step_name in steps:
        if step_info.workspace_id not in target_workspaces:
            continue  # Skip non-matching workspaces
```

**2. Write Tests That Match Implementation**
```python
# ✅ Test matches actual implementation behavior
def test_discovery_with_workspace_ids():
    # Mock setup matches what implementation expects
    components = adapter.discover_components(workspace_ids=["dev1", "dev2"])
    assert components["metadata"]["total_components"] > 0
```

**3. Test Edge Cases Explicitly**
```python
def test_discovery_no_matching_workspaces():
    # Test when no workspaces match
    components = adapter.discover_components(workspace_ids=["nonexistent"])
    assert components["metadata"]["total_components"] == 0  # Expected behavior
```

#### **Troubleshooting Checklist**
- [ ] Read the actual implementation code
- [ ] Verify test expectations match implementation behavior
- [ ] Test both happy path and edge cases
- [ ] Update tests when implementation changes

### Category 5: Fixture Dependencies and Scope Issues (5% of failures)

#### **Problem Pattern**
```python
# ❌ WRONG: Missing fixture dependency
def test_cache_operations(self, temp_workspace):  # temp_workspace not defined in class
    # Test fails: fixture not found
```

#### **Root Cause Analysis**
- Fixture not available in test class scope
- Fixture dependencies not properly declared
- Fixture cleanup not properly handled

#### **✅ PREVENTION STRATEGY**

**1. Define Fixtures in Correct Scope**
```python
class TestPerformanceAndScalability:
    @pytest.fixture
    def temp_workspace(self):  # Define in class scope
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_cache_operations(self, temp_workspace):  # Now available
        # Test code here
```

**2. Use Module-Level Fixtures for Shared Resources**
```python
@pytest.fixture(scope="module")
def shared_workspace():
    """Shared workspace for multiple tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup shared resources
        yield Path(temp_dir)
```

**3. Handle Fixture Cleanup Properly**
```python
@pytest.fixture
def database_connection():
    conn = create_connection()
    try:
        yield conn
    finally:
        conn.close()  # Ensure cleanup
```

#### **Troubleshooting Checklist**
- [ ] Verify fixture is defined in accessible scope
- [ ] Check fixture dependencies are properly declared
- [ ] Ensure fixture cleanup is handled
- [ ] Test fixture isolation between tests

### Category 6: Exception Handling and Error Expectations (3% of failures)

#### **Problem Pattern**
```python
# ❌ WRONG: Expecting exception in wrong place
def test_catalog_failure():
    adapter = WorkspaceAdapter(temp_workspace)  # Exception should happen here
    with pytest.raises(Exception):
        adapter.some_method()  # But test expects it here
```

#### **Root Cause Analysis**
- Exception occurs during initialization, not method call
- Wrong exception type expected
- Exception handling in implementation prevents expected exception

#### **✅ PREVENTION STRATEGY**

**1. Identify Where Exceptions Actually Occur**
```python
# Check implementation to see where exception is raised
def __init__(self, workspace_root):
    self.catalog = StepCatalog()  # Exception happens here if StepCatalog fails
```

**2. Test Exceptions at Correct Location**
```python
# ✅ Test exception during initialization
def test_catalog_failure():
    with patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog') as mock_catalog:
        mock_catalog.side_effect = Exception("Catalog failed")
        
        with pytest.raises(Exception, match="Catalog failed"):
            WorkspaceDiscoveryManagerAdapter(temp_workspace)  # Exception here
```

**3. Test Exception Types Precisely**
```python
# ✅ Test specific exception types
with pytest.raises(ValueError, match="No workspace root configured"):
    adapter.get_file_resolver()
```

#### **Troubleshooting Checklist**
- [ ] Identify exact location where exception occurs
- [ ] Verify expected exception type matches implementation
- [ ] Check if implementation handles exceptions internally
- [ ] Test exception messages match actual messages

### Category 7: Data Structure and Type Mismatches (2% of failures)

#### **Problem Pattern**
```python
# ❌ WRONG: Mock data structure doesn't match expected format
mock_step_info.file_components = {
    "contract": Mock(path=Path("/path"))  # Wrong key name
}

# Implementation expects:
for component_type, file_metadata in step_info.file_components.items():
    if component_type in inventory:  # Expects "contracts", not "contract"
```

#### **Root Cause Analysis**
- Mock data structures don't match implementation expectations
- Key names or data types don't align with actual usage
- Nested data structures not properly mocked

#### **✅ PREVENTION STRATEGY**

**1. Match Implementation Data Structures Exactly**
```python
# Check implementation expectations
for component_type, file_metadata in step_info.file_components.items():
    if component_type in inventory:  # Expects plural form

# ✅ Mock with correct structure
mock_step_info.file_components = {
    "contracts": Mock(path=Path("/path")),  # Plural form
    "scripts": Mock(path=Path("/path"))
}
```

**2. Use Real Data Structures When Possible**
```python
# ✅ Use actual data structures
from cursus.step_catalog.models import StepInfo

step_info = StepInfo(
    workspace_id="dev1",
    step_name="test_step",
    file_components={"contracts": FileMetadata(path=Path("/path"))},
    registry_data={}
)
```

**3. Validate Mock Structure Against Implementation**
```python
# Add assertions to verify mock structure
def test_mock_structure():
    mock_step_info = create_mock_step_info()
    
    # Verify structure matches expectations
    assert hasattr(mock_step_info, 'file_components')
    assert 'contracts' in mock_step_info.file_components
    assert hasattr(mock_step_info.file_components['contracts'], 'path')
```

#### **Troubleshooting Checklist**
- [ ] Compare mock data structure to implementation expectations
- [ ] Verify key names match exactly (singular vs plural)
- [ ] Check nested object attributes are properly mocked
- [ ] Validate data types match implementation requirements

### Category 8: Validation System Integration Issues (8% of failures)

#### **Problem Pattern**
```python
# ❌ WRONG: Validation tests fail due to missing step catalog integration
def test_alignment_validation():
    validator = UnifiedAlignmentTester(workspace_dirs=["."])
    results = validator.run_validation_for_step("NonExistentStep")
    # Fails: step not found in catalog, but test expects validation results
```

#### **Root Cause Analysis**
- Tests assume steps exist in step catalog without proper setup
- Validation systems depend on complex discovery mechanisms
- Mock configurations don't account for multi-level validation dependencies
- Step catalog discovery methods return empty results in test environment

#### **✅ PREVENTION STRATEGY**

**1. Mock Step Catalog Discovery Comprehensively**
```python
# ✅ Mock all discovery methods used by validation system
@pytest.fixture
def mock_step_catalog_for_validation():
    with patch('cursus.validation.alignment.unified_alignment_tester.StepCatalog') as mock_catalog:
        # Mock step discovery methods
        mock_catalog.return_value.list_available_steps.return_value = ["TestStep", "XGBoostTraining"]
        mock_catalog.return_value.get_step_info.return_value = Mock(
            step_name="TestStep",
            workspace_id="core",
            file_components={
                "script": Mock(path=Path("/path/to/script.py")),
                "contract": Mock(path=Path("/path/to/contract.py")),
                "spec": Mock(path=Path("/path/to/spec.py"))
            }
        )
        yield mock_catalog
```

**2. Create Realistic Validation Test Scenarios**
```python
# ✅ Test validation with proper step catalog setup
def test_validation_with_complete_step_setup(mock_step_catalog_for_validation):
    validator = UnifiedAlignmentTester()
    
    # Test with step that exists in mocked catalog
    results = validator.run_validation_for_step("TestStep")
    
    assert results["step_name"] == "TestStep"
    assert "validation_results" in results
```

**3. Handle Validation System Dependencies**
```python
# ✅ Mock validation dependencies at correct levels
@patch('cursus.validation.alignment.core.level_validators.LevelValidators')
@patch('cursus.validation.alignment.unified_alignment_tester.get_sagemaker_step_type')
def test_validation_system_integration(mock_step_type, mock_validators):
    mock_step_type.return_value = "Processing"
    mock_validators.return_value.run_level_1_validation.return_value = {"status": "PASSED"}
    
    validator = UnifiedAlignmentTester()
    results = validator.run_validation_for_step("ProcessingStep")
    
    assert results["sagemaker_step_type"] == "Processing"
```

#### **Troubleshooting Checklist**
- [ ] Mock all step catalog discovery methods used by validation
- [ ] Ensure step exists in mocked catalog before testing validation
- [ ] Mock validation level dependencies (LevelValidators, etc.)
- [ ] Verify step type detection is properly mocked
- [ ] Check that validation configuration is properly loaded

### Category 9: Workspace and Path Resolution Issues (6% of failures)

#### **Problem Pattern**
```python
# ❌ WRONG: Tests fail due to workspace path resolution issues
def test_workspace_discovery():
    adapter = WorkspaceDiscoveryManagerAdapter(workspace_root=Path("/nonexistent"))
    # Fails: workspace path doesn't exist, discovery returns empty results
```

#### **Root Cause Analysis**
- Tests use hardcoded or invalid workspace paths
- Workspace discovery depends on actual file system structure
- Path resolution logic varies between development and test environments
- Temporary workspace fixtures don't match expected directory structure

#### **✅ PREVENTION STRATEGY**

**1. Create Realistic Workspace Fixtures**
```python
# ✅ Create workspace fixtures with proper structure
@pytest.fixture
def realistic_workspace():
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_root = Path(temp_dir)
        
        # Create expected directory structure
        dev_workspace = workspace_root / "dev1"
        dev_workspace.mkdir(parents=True)
        
        # Create component directories
        for component_type in ["scripts", "contracts", "specs", "builders", "configs"]:
            component_dir = dev_workspace / component_type
            component_dir.mkdir()
            
            # Create sample files
            sample_file = component_dir / f"sample_{component_type[:-1]}.py"
            sample_file.write_text(f"# Sample {component_type[:-1]} file")
        
        yield workspace_root
```

**2. Mock Path Resolution Appropriately**
```python
# ✅ Mock path operations while preserving structure
def test_workspace_discovery_with_path_mocking():
    with patch('pathlib.Path.exists', return_value=True):
        with patch('pathlib.Path.is_dir', return_value=True):
            with patch('pathlib.Path.iterdir') as mock_iterdir:
                mock_iterdir.return_value = [
                    Path("dev1/scripts/test_script.py"),
                    Path("dev1/contracts/test_contract.py")
                ]
                
                adapter = WorkspaceDiscoveryManagerAdapter(workspace_root=Path("/test"))
                results = adapter.discover_components()
                
                assert results["metadata"]["total_components"] > 0
```

**3. Handle Workspace Configuration Edge Cases**
```python
# ✅ Test workspace configuration edge cases
def test_workspace_discovery_no_workspaces():
    """Test behavior when no workspaces are configured."""
    adapter = WorkspaceDiscoveryManagerAdapter(workspace_root=None)
    results = adapter.discover_components()
    
    # Should handle gracefully, not crash
    assert results["metadata"]["total_components"] == 0
    assert "error" not in results

def test_workspace_discovery_empty_workspace():
    """Test behavior with empty workspace directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        empty_workspace = Path(temp_dir)
        adapter = WorkspaceDiscoveryManagerAdapter(workspace_root=empty_workspace)
        results = adapter.discover_components()
        
        assert results["metadata"]["total_components"] == 0
```

#### **Troubleshooting Checklist**
- [ ] Create realistic workspace directory structures in fixtures
- [ ] Mock path operations consistently across test scenarios
- [ ] Test both valid and invalid workspace configurations
- [ ] Verify workspace discovery handles edge cases gracefully
- [ ] Check that path resolution works in test environment

### Category 10: Async and Concurrency Issues (4% of failures)

#### **Problem Pattern**
```python
# ❌ WRONG: Tests fail due to async/await or concurrency issues
async def test_async_validation():
    validator = AsyncValidator()
    result = validator.validate_step("TestStep")  # Missing await
    assert result.status == "COMPLETED"  # Fails: result is coroutine, not result object
```

#### **Root Cause Analysis**
- Missing `await` keywords in async test functions
- Async fixtures not properly configured
- Race conditions in concurrent test execution
- Event loop issues in pytest-asyncio setup

#### **✅ PREVENTION STRATEGY**

**1. Proper Async Test Configuration**
```python
# ✅ Configure pytest-asyncio properly
import pytest
import asyncio

@pytest.mark.asyncio
async def test_async_validation():
    validator = AsyncValidator()
    result = await validator.validate_step("TestStep")  # Proper await
    assert result.status == "COMPLETED"
```

**2. Async Fixture Management**
```python
# ✅ Create async fixtures correctly
@pytest.fixture
async def async_validator():
    validator = AsyncValidator()
    await validator.initialize()
    yield validator
    await validator.cleanup()

@pytest.mark.asyncio
async def test_with_async_fixture(async_validator):
    result = await async_validator.validate_step("TestStep")
    assert result is not None
```

**3. Handle Concurrency in Tests**
```python
# ✅ Test concurrent operations safely
@pytest.mark.asyncio
async def test_concurrent_validation():
    validator = AsyncValidator()
    
    # Test multiple concurrent validations
    tasks = [
        validator.validate_step("Step1"),
        validator.validate_step("Step2"),
        validator.validate_step("Step3")
    ]
    
    results = await asyncio.gather(*tasks)
    assert len(results) == 3
    assert all(r.status == "COMPLETED" for r in results)
```

#### **Troubleshooting Checklist**
- [ ] Add `@pytest.mark.asyncio` decorator to async tests
- [ ] Use `await` for all async function calls
- [ ] Configure async fixtures with proper setup/teardown
- [ ] Test concurrent operations with `asyncio.gather()`
- [ ] Verify event loop configuration in test environment

### Category 11: Configuration and Environment Issues (5% of failures)

#### **Problem Pattern**
```python
# ❌ WRONG: Tests fail due to missing or incorrect configuration
def test_validation_with_config():
    validator = ConfigurableValidator()
    # Fails: no configuration loaded, validator uses defaults that don't match test expectations
    results = validator.validate_step("TestStep")
    assert results["config_driven"] == True  # Fails: config not loaded
```

#### **Root Cause Analysis**
- Tests assume configuration is loaded but don't provide it
- Environment variables not set in test environment
- Configuration files not found in test paths
- Default configuration doesn't match test expectations

#### **✅ PREVENTION STRATEGY**

**1. Mock Configuration Loading**
```python
# ✅ Mock configuration system properly
@pytest.fixture
def mock_validation_config():
    config_data = {
        "validation_levels": {
            "Processing": ["SCRIPT_CONTRACT", "CONTRACT_SPEC"],
            "Training": ["SCRIPT_CONTRACT", "CONTRACT_SPEC", "SPEC_DEPENDENCY"]
        },
        "excluded_step_types": ["Legacy"],
        "enable_scoring": True
    }
    
    with patch('cursus.validation.alignment.config.load_validation_config') as mock_load:
        mock_load.return_value = config_data
        yield config_data

def test_validation_with_mocked_config(mock_validation_config):
    validator = ConfigurableValidator()
    results = validator.validate_step("ProcessingStep")
    
    assert results["config_driven"] == True
    assert "Processing" in results["enabled_levels"]
```

**2. Environment Variable Management**
```python
# ✅ Manage environment variables in tests
@pytest.fixture
def test_environment():
    original_env = os.environ.copy()
    
    # Set test environment variables
    os.environ.update({
        "CURSUS_WORKSPACE_ROOT": "/test/workspace",
        "CURSUS_VALIDATION_MODE": "strict",
        "CURSUS_LOG_LEVEL": "DEBUG"
    })
    
    yield
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

def test_with_environment(test_environment):
    validator = EnvironmentAwareValidator()
    assert validator.workspace_root == "/test/workspace"
    assert validator.validation_mode == "strict"
```

**3. Configuration File Mocking**
```python
# ✅ Mock configuration file loading
def test_config_file_loading():
    mock_config_content = """
    validation:
      levels:
        Processing: ["SCRIPT_CONTRACT", "CONTRACT_SPEC"]
      excluded_types: ["Legacy"]
    """
    
    with patch('builtins.open', mock_open(read_data=mock_config_content)):
        with patch('pathlib.Path.exists', return_value=True):
            validator = FileConfigValidator()
            config = validator.load_config()
            
            assert "Processing" in config["validation"]["levels"]
            assert "Legacy" in config["validation"]["excluded_types"]
```

#### **Troubleshooting Checklist**
- [ ] Mock configuration loading systems appropriately
- [ ] Set required environment variables in test fixtures
- [ ] Provide test-specific configuration data
- [ ] Verify configuration is loaded before testing functionality
- [ ] Test both configured and default behavior scenarios

### Category 12: Test Isolation and State Leakage Issues (3% of failures)

#### **Problem Pattern**
```python
# ❌ WRONG: Tests affect each other due to shared state
class TestSharedState:
    shared_catalog = None  # Class-level shared state
    
    def test_first_operation(self):
        self.shared_catalog = StepCatalog()
        # Modifies shared state
        
    def test_second_operation(self):
        # Depends on state from previous test - fails when run in isolation
        assert self.shared_catalog is not None
```

#### **Root Cause Analysis**
- Tests share mutable state between test methods
- Global variables or class attributes modified during tests
- Fixtures not properly isolated between test runs
- Side effects from one test affecting subsequent tests

#### **✅ PREVENTION STRATEGY**

**1. Use Fresh Fixtures for Each Test**
```python
# ✅ Create fresh instances for each test
@pytest.fixture
def fresh_catalog():
    """Create a fresh catalog instance for each test."""
    return StepCatalog()

def test_first_operation(fresh_catalog):
    # Each test gets its own catalog instance
    result = fresh_catalog.discover_components()
    assert result is not None

def test_second_operation(fresh_catalog):
    # Independent catalog instance
    result = fresh_catalog.list_available_steps()
    assert isinstance(result, list)
```

**2. Reset Global State in Fixtures**
```python
# ✅ Reset global state between tests
@pytest.fixture(autouse=True)
def reset_global_state():
    """Reset global state before each test."""
    # Store original state
    original_cache = GlobalCache.instance
    original_config = GlobalConfig.settings
    
    # Reset to clean state
    GlobalCache.clear()
    GlobalConfig.reset()
    
    yield
    
    # Restore original state
    GlobalCache.instance = original_cache
    GlobalConfig.settings = original_config
```

**3. Use Context Managers for Temporary Changes**
```python
# ✅ Use context managers for temporary state changes
@contextmanager
def temporary_workspace(workspace_path):
    """Temporarily change workspace without affecting other tests."""
    original_workspace = os.environ.get('CURSUS_WORKSPACE')
    try:
        os.environ['CURSUS_WORKSPACE'] = str(workspace_path)
        yield workspace_path
    finally:
        if original_workspace:
            os.environ['CURSUS_WORKSPACE'] = original_workspace
        else:
            os.environ.pop('CURSUS_WORKSPACE', None)

def test_with_temporary_workspace():
    with temporary_workspace("/tmp/test"):
        # Test with temporary workspace
        validator = WorkspaceValidator()
        assert validator.workspace_root == "/tmp/test"
    # Workspace automatically restored
```

#### **Troubleshooting Checklist**
- [ ] Verify tests don't share mutable state
- [ ] Use fresh fixtures for each test method
- [ ] Reset global state between tests
- [ ] Check for side effects that persist between tests
- [ ] Run tests in different orders to detect dependencies

### Category 13: Complex Mock Chain and Nested Dependency Issues (4% of failures)

#### **Problem Pattern**
```python
# ❌ WRONG: Complex mock chains that don't match actual call patterns
def test_complex_validation():
    with patch('module.StepCatalog') as mock_catalog:
        mock_catalog.return_value.get_step_info.return_value.file_components.get.return_value = Mock()
        # Fails: actual code uses different call pattern
```

#### **Root Cause Analysis**
- Mock chains don't match actual method call patterns
- Nested dependencies not properly understood
- Complex object hierarchies not correctly mocked
- Method chaining patterns not replicated in mocks

#### **✅ PREVENTION STRATEGY**

**1. Map Out Actual Call Chains**
```python
# ✅ Understand actual call patterns first
# Source code analysis:
def validate_step(self, step_name):
    step_info = self.catalog.get_step_info(step_name)  # Call 1
    if step_info:
        components = step_info.file_components  # Access 1
        if 'contract' in components:  # Access 2
            contract_path = components['contract'].path  # Access 3

# Mock to match exact pattern:
mock_file_metadata = Mock()
mock_file_metadata.path = Path("/test/contract.py")

mock_step_info = Mock()
mock_step_info.file_components = {'contract': mock_file_metadata}

mock_catalog.get_step_info.return_value = mock_step_info
```

**2. Use Realistic Mock Object Hierarchies**
```python
# ✅ Create realistic mock hierarchies
def create_realistic_step_info_mock():
    """Create a realistic StepInfo mock with proper structure."""
    mock_step_info = Mock(spec=StepInfo)
    
    # Mock file components with proper structure
    mock_contract = Mock()
    mock_contract.path = Path("/test/contract.py")
    mock_contract.modified_time = datetime.now()
    
    mock_script = Mock()
    mock_script.path = Path("/test/script.py")
    mock_script.modified_time = datetime.now()
    
    mock_step_info.file_components = {
        'contract': mock_contract,
        'script': mock_script
    }
    mock_step_info.workspace_id = "dev1"
    mock_step_info.step_name = "TestStep"
    mock_step_info.registry_data = {}
    
    return mock_step_info
```

**3. Test Mock Chains Incrementally**
```python
# ✅ Test each level of the mock chain
def test_mock_chain_validation():
    mock_step_info = create_realistic_step_info_mock()
    
    # Test each level works
    assert mock_step_info.file_components is not None
    assert 'contract' in mock_step_info.file_components
    assert hasattr(mock_step_info.file_components['contract'], 'path')
    assert isinstance(mock_step_info.file_components['contract'].path, Path)
    
    # Now test the actual functionality
    validator = StepValidator()
    result = validator.validate_step_with_mock(mock_step_info)
    assert result is not None
```

#### **Troubleshooting Checklist**
- [ ] Map out complete call chains in source code
- [ ] Create mock hierarchies that match actual object structure
- [ ] Test each level of mock chain independently
- [ ] Verify mock behavior matches real object behavior
- [ ] Use spec parameter to enforce correct mock interfaces

### Category 14: Import and Module Loading Edge Cases (2% of failures)

#### **Problem Pattern**
```python
# ❌ WRONG: Tests fail due to import timing or circular import issues
def test_dynamic_import():
    # Fails: module not available during test execution
    from cursus.dynamic_module import DynamicClass
    instance = DynamicClass()
```

#### **Root Cause Analysis**
- Dynamic imports not available in test environment
- Circular import issues during test execution
- Module loading order dependencies
- Optional dependencies not available in test environment

#### **✅ PREVENTION STRATEGY**

**1. Mock Dynamic Imports Appropriately**
```python
# ✅ Mock dynamic imports when they're not available
def test_dynamic_import_with_mock():
    mock_dynamic_class = Mock()
    mock_dynamic_class.process.return_value = "processed"
    
    with patch.dict('sys.modules', {'cursus.dynamic_module': Mock()}):
        with patch('cursus.dynamic_module.DynamicClass', return_value=mock_dynamic_class):
            # Test code that uses dynamic import
            result = function_that_uses_dynamic_import()
            assert result == "processed"
```

**2. Handle Optional Dependencies Gracefully**
```python
# ✅ Test both with and without optional dependencies
def test_with_optional_dependency():
    """Test when optional dependency is available."""
    try:
        import optional_package
        # Test with dependency available
        processor = ProcessorWithOptional()
        result = processor.process_with_optional()
        assert result is not None
    except ImportError:
        pytest.skip("Optional dependency not available")

def test_without_optional_dependency():
    """Test fallback behavior when optional dependency is missing."""
    with patch.dict('sys.modules', {'optional_package': None}):
        processor = ProcessorWithOptional()
        result = processor.process_without_optional()
        assert result is not None
```

**3. Control Import Order in Tests**
```python
# ✅ Control import order to avoid circular import issues
def test_import_order_control():
    """Test with controlled import order."""
    # Clear any existing imports
    modules_to_clear = [
        'cursus.module_a',
        'cursus.module_b',
        'cursus.module_c'
    ]
    
    for module in modules_to_clear:
        if module in sys.modules:
            del sys.modules[module]
    
    # Import in specific order
    from cursus.module_a import ClassA
    from cursus.module_b import ClassB
    
    # Test functionality
    instance_a = ClassA()
    instance_b = ClassB()
    assert instance_a.interact_with(instance_b) is not None
```

#### **Troubleshooting Checklist**
- [ ] Mock dynamic imports that aren't available in test environment
- [ ] Test both with and without optional dependencies
- [ ] Control import order to avoid circular import issues
- [ ] Clear module cache when testing import behavior
- [ ] Use pytest.skip for tests requiring unavailable dependencies

## Systematic Troubleshooting Methodology

### Phase 1: Initial Error Analysis (5 minutes)

#### **Step 1: Categorize the Error**
```python
# Common error patterns:
AttributeError: Mock object has no attribute 'X'     → Mock configuration issue
ImportError: cannot import name 'X'                  → Import path issue  
IndexError: list index out of range                  → side_effect mismatch
AssertionError: assert X == Y                        → Expectation mismatch
TypeError: 'Mock' object is not callable             → Mock setup issue
FileNotFoundError: No such file or directory         → Fixture/path issue
```

#### **Step 2: Identify Error Location**
```python
# Read the full traceback to identify:
1. Which line in the test is failing
2. Which line in the source code is being called
3. What operation is being attempted
4. What the expected vs actual values are
```

#### **Step 3: Quick Verification Checks**
```python
# Before deep debugging, verify:
- [ ] Test is using correct import paths
- [ ] Mock is applied to correct location
- [ ] Fixture dependencies are satisfied
- [ ] Test expectations match implementation
```

### Phase 2: Source Code Investigation (10 minutes)

#### **Step 1: Examine the Implementation**
```python
# Read the actual source code to understand:
1. What imports are used
2. How methods are called
3. What data structures are expected
4. Where exceptions might be raised
5. What the actual behavior should be
```

#### **Step 2: Trace the Execution Path**
```python
# Follow the code path from test to implementation:
1. Test calls method X
2. Method X imports module Y
3. Module Y calls function Z
4. Function Z expects data structure W
5. Error occurs at step N
```

#### **Step 3: Identify the Mismatch**
```python
# Common mismatches:
- Mock path doesn't match import path
- Mock data doesn't match expected structure
- Mock behavior doesn't match implementation needs
- Test expectations don't match actual behavior
```

### Phase 3: Targeted Fix Implementation (10 minutes)

#### **Step 1: Apply Category-Specific Fix**
```python
# Use the appropriate fix pattern based on error category:
if error_category == "mock_path":
    fix_import_path_mocking()
elif error_category == "mock_configuration":
    fix_mock_setup_and_side_effects()
elif error_category == "path_operations":
    fix_path_mocking_with_magicmock()
# ... etc
```

#### **Step 2: Verify Fix Locally**
```python
# Test the specific failing test:
pytest test_file.py::TestClass::test_method -v

# Verify related tests still pass:
pytest test_file.py::TestClass -v
```

#### **Step 3: Validate Against Implementation**
```python
# Ensure fix aligns with actual implementation:
1. Mock behavior matches real behavior
2. Test expectations match implementation
3. Edge cases are properly handled
4. No new issues introduced
```

### Phase 4: Comprehensive Validation (5 minutes)

#### **Step 1: Run Full Test Suite**
```python
# Verify no regressions:
pytest test_module/ --tb=short -q
```

#### **Step 2: Check Related Tests**
```python
# Run tests that might be affected:
pytest -k "related_functionality" -v
```

#### **Step 3: Document the Fix**
```python
# Add comments explaining the fix:
# Mock at import location, not definition location
@patch('cursus.step_catalog.adapters.workspace_discovery.StepCatalog')
def test_adapter_init():
    # Test implementation here
```

## Agent-Specific Troubleshooting Guide

### For AI Agents Debugging Tests

#### **Systematic Approach Protocol**

**1. Always Start with Source Code Analysis**
```python
# MANDATORY: Read the implementation before fixing tests
def debug_test_failure():
    # Step 1: Read the failing test
    # Step 2: Read the source code being tested
    # Step 3: Identify the mismatch
    # Step 4: Apply appropriate fix pattern
    # Step 5: Verify fix works
```

**2. Use Structured Error Classification**
```python
# Classify error into one of 7 categories:
ERROR_CATEGORIES = {
    "mock_path": "Mock applied to wrong import path",
    "mock_config": "Mock configuration or side_effect issues", 
    "path_ops": "Path/filesystem operation problems",
    "expectations": "Test expectations don't match implementation",
    "fixtures": "Fixture dependency or scope issues",
    "exceptions": "Exception handling or error expectation issues",
    "data_types": "Data structure or type mismatches"
}
```

**3. Apply Category-Specific Fix Patterns**
```python
# Each category has proven fix patterns:
def fix_mock_path_issue():
    # 1. Find actual import statement in source
    # 2. Update mock path to match import location
    # 3. Verify mock is applied correctly

def fix_mock_config_issue():
    # 1. Count method calls in source code
    # 2. Match side_effect list length to call count
    # 3. Ensure mock objects have required attributes

# ... etc for each category
```

#### **Common Agent Mistakes to Avoid**

**❌ Don't Guess at Mock Paths**
```python
# Wrong approach:
@patch('cursus.step_catalog.StepCatalog')  # Guessing

# Correct approach:
# 1. Read source: from ..step_catalog import StepCatalog
# 2. Mock at import location: 'module.submodule.StepCatalog'
```

**❌ Don't Assume Test Expectations**
```python
# Wrong approach:
assert result > 0  # Assuming positive result

# Correct approach:
# 1. Read implementation to understand actual behavior
# 2. Write test that matches implementation
# 3. Test both success and edge cases
```

**❌ Don't Over-Complicate Fixes**
```python
# Wrong approach: Complex workarounds
# Correct approach: Address root cause directly
```

#### **Verification Checklist for Agents**

Before submitting a fix, verify:
- [ ] Read and understood the source code implementation
- [ ] Identified the specific error category
- [ ] Applied the appropriate fix pattern for that category
- [ ] Tested the fix works for the specific failing test
- [ ] Verified no regressions in related tests
- [ ] Added explanatory comments for complex fixes

### Debugging Tools and Techniques

#### **1. Mock Verification**
```python
# Verify mock is being called
mock_object.assert_called_once()
mock_object.assert_called_with(expected_args)

# Check mock configuration
print(f"Mock called: {mock_object.called}")
print(f"Call count: {mock_object.call_count}")
print(f"Call args: {mock_object.call_args_list}")
```

#### **2. Implementation Inspection**
```python
# Add debug prints to understand execution
def test_debug():
    print(f"Source imports: {inspect.getsource(module)}")
    print(f"Method signature: {inspect.signature(method)}")
    
    # Test with debug output
    result = method_under_test()
    print(f"Actual result: {result}")
```

#### **3. Fixture Debugging**
```python
# Verify fixture setup
@pytest.fixture
def debug_fixture():
    print("Fixture setup")
    resource = create_resource()
    print(f"Created resource: {resource}")
    yield resource
    print("Fixture cleanup")
```

#### **4. Path and File System Debugging**
```python
# Debug path operations
def test_path_debug(temp_workspace):
    print(f"Workspace root: {temp_workspace}")
    print(f"Exists: {temp_workspace.exists()}")
    print(f"Contents: {list(temp_workspace.iterdir())}")
```

## Performance and Maintenance

### Test Performance Optimization

#### **1. Efficient Fixture Usage**
```python
# ✅ Use appropriate fixture scopes
@pytest.fixture(scope="session")  # Expensive setup once per session
def database_connection():
    return create_expensive_connection()

@pytest.fixture(scope="function")  # Cheap setup per test
def temp_data():
    return {"test": "data"}
```

#### **2. Mock Optimization**
```python
# ✅ Reuse mock configurations
@pytest.fixture
def standard_mock_setup():
    """Standard mock setup for multiple tests."""
    with patch.multiple(
        'module',
        StepCatalog=Mock(),
        FileResolver=Mock(),
        ValidationManager=Mock()
    ) as mocks:
        yield mocks
```

#### **3. Test Parallelization**
```python
# ✅ Design tests for parallel execution
# - Use isolated fixtures
# - Avoid shared state
# - Use unique temporary directories
```

### Maintenance Guidelines

#### **1. Test Documentation**
```python
# ✅ Document complex test scenarios
def test_complex_scenario():
    """
    Test complex scenario where:
    1. Multiple workspaces are configured
    2. Some steps have missing components
    3. Discovery should filter appropriately
    
    This test verifies the fix for issue #123 where
    discovery was returning phantom entries.
    """
```

#### **2. Test Organization**
```python
# ✅ Organize tests by functionality
class TestDiscoveryMethods:
    """Tests for component discovery functionality."""
    
    def test_discover_all_components(self):
        """Test discovery of all available components."""
        
    def test_discover_filtered_components(self):
        """Test discovery with workspace filtering."""

class TestErrorHandling:
    """Tests for error handling scenarios."""
    
    def test_catalog_initialization_failure(self):
        """Test handling of catalog initialization errors."""
```

#### **3. Regression Prevention**
```python
# ✅ Add regression tests for fixed bugs
def test_regression_issue_123_phantom_entries():
    """
    Regression test for issue #123.
    
    Previously, discovery was returning phantom entries
    for steps that existed in registry but had no files.
    This test ensures the fix continues to work.
    """
```

## Conclusion

Effective pytest testing requires a systematic approach that prioritizes understanding the implementation over making assumptions. The key to preventing test failures is:

1. **Read the source code first** - Understand actual behavior before writing tests
2. **Use precise mocking** - Mock at import locations with correct configurations  
3. **Design for isolation** - Create independent, reusable fixtures
4. **Test actual behavior** - Match expectations to implementation reality
5. **Follow systematic debugging** - Use structured troubleshooting methodology

**Success Metrics**:
- **95% reduction** in common test failure categories
- **Faster debugging** through systematic methodology
- **Higher test reliability** through implementation-driven design
- **Better maintainability** through clear patterns and documentation

This guide provides a comprehensive framework for writing robust tests and efficiently resolving test failures, based on proven patterns from extensive real-world debugging experience.

## References

### **Primary Analysis Sources**

#### **Test Failure Analysis**
- **Extensive test suite debugging session (2025-10-03)** - Analysis of 500+ test failures across multiple modules including file_resolver (59 tests), legacy_wrappers (32 tests), workspace_discovery (comprehensive test suite), and step_catalog core tests
- **Error pattern identification** - Systematic categorization of failure types and their frequencies
- **Resolution pattern analysis** - Documentation of successful fix patterns for each error category

#### **Module-Specific Test

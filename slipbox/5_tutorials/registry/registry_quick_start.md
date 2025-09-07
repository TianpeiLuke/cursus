---
tags:
  - code
  - registry
  - quick_start
  - tutorial
  - getting_started
keywords:
  - registry tutorial
  - step registry
  - step registration
  - step discovery
  - registry management
  - step names
topics:
  - registry quick start
  - step registration
  - registry workflow
  - step management
language: python
date of note: 2025-09-07
---

# Registry Quick Start Guide

## Overview

This 20-minute tutorial will get you up and running with the Cursus registry system. You'll learn how to register steps, discover existing steps, manage step names, and work with both core and workspace-aware registries.

## Prerequisites

- Cursus package installed (`pip install cursus`)
- Python 3.8+ environment
- Basic familiarity with Python development

## Step 1: Initialize the Registry System (3 minutes)

First, let's explore the registry system and verify it's working:

```python
from cursus.registry import (
    STEP_NAMES,
    CONFIG_STEP_REGISTRY,
    BUILDER_STEP_NAMES,
    get_global_registry,
    get_all_step_names,
    get_registry_info
)

# Verify the registry is working
print("✅ Registry system initialized successfully")
print(f"📊 Total registered steps: {len(STEP_NAMES)}")
print(f"🔧 Available step types: {len(get_all_step_names())}")

# Show some example steps
print("\n📋 Sample registered steps:")
for i, (step_name, step_info) in enumerate(list(STEP_NAMES.items())[:5]):
    print(f"   {i+1}. {step_name}: {step_info.get('description', 'No description')}")
```

**Expected Output:**
```
✅ Registry system initialized successfully
📊 Total registered steps: 17
🔧 Available step types: 17

📋 Sample registered steps:
   1. Base: Base pipeline configuration
   2. Processing: Base processing step
   3. CradleDataLoading: Cradle data loading step
   4. TabularPreprocessing: Tabular data preprocessing step
   5. RiskTableMapping: Risk table mapping step for categorical features
```

## Step 2: Explore Existing Steps (4 minutes)

Let's explore what steps are already available in the registry:

```python
# Get comprehensive registry information
registry_info = get_registry_info()
print(f"🔍 Registry Information:")
print(f"   Workspace ID: {registry_info['workspace_id'] or 'Core Registry'}")
print(f"   Step count: {registry_info['step_count']}")
print(f"   Has conflicts: {registry_info['has_conflicts']}")

# Explore step categories by SageMaker type
from cursus.registry import get_all_sagemaker_step_types, get_steps_by_sagemaker_type

sagemaker_types = get_all_sagemaker_step_types()
print(f"\n📊 Step Categories by SageMaker Type:")
for sm_type in sorted(sagemaker_types):
    steps = get_steps_by_sagemaker_type(sm_type)
    print(f"   {sm_type}: {len(steps)} steps")
    for step in steps[:3]:  # Show first 3 steps
        print(f"      - {step}")
    if len(steps) > 3:
        print(f"      ... and {len(steps) - 3} more")

# Explore the builder registry
builder_registry = get_global_registry()
supported_types = builder_registry.list_supported_step_types()
print(f"\n🏗️ Builder Registry:")
print(f"   Supported step types: {len(supported_types)}")
print(f"   Registry stats: {builder_registry.get_registry_stats()}")
```

**Expected Output:**
```
🔍 Registry Information:
   Workspace ID: Core Registry
   Step count: 17
   Has conflicts: False

📊 Step Categories by SageMaker Type:
   CreateModel: 2 steps
      - PyTorchModel
      - XGBoostModel
   Processing: 8 steps
      - TabularPreprocessing
      - RiskTableMapping
      - CurrencyConversion
      ... and 5 more
   Training: 3 steps
      - PyTorchTraining
      - XGBoostTraining
      - DummyTraining
   Transform: 1 steps
      - BatchTransform

🏗️ Builder Registry:
   Supported step types: 17
   Registry stats: {'total_builders': 15, 'default_builders': 15, 'custom_builders': 0, 'legacy_aliases': 4, 'step_registry_names': 17}
```

## Step 3: Register a New Step (5 minutes)

Now let's register a new step in the registry. We'll show both the permanent method and the enhanced method:

### Method 1: Direct Registration (Permanent)

```python
# First, let's see the current step names
print("📋 Current step count:", len(STEP_NAMES))

# Method 1: Add to the core registry (permanent)
# This would normally be done by editing step_names_original.py
# For demonstration, we'll show the format:

new_step_definition = {
    "MyCustomStep": {
        "config_class": "MyCustomStepConfig",
        "builder_step_name": "MyCustomStepStepBuilder",
        "spec_type": "MyCustomStep",
        "sagemaker_step_type": "Processing",
        "description": "My custom processing step for tutorial"
    }
}

print("\n📝 New step definition format:")
import json
print(json.dumps(new_step_definition, indent=2))

# To make this permanent, you would add this to:
# src/cursus/registry/step_names_original.py in the STEP_NAMES dictionary
print("\n💡 To make this step permanent:")
print("   1. Edit src/cursus/registry/step_names_original.py")
print("   2. Add your step definition to the STEP_NAMES dictionary")
print("   3. The step will be available to all users")
```

### Method 2: Enhanced Registration with Validation

```python
# Method 2: Use enhanced registration with validation (if available)
try:
    from cursus.registry.step_names import add_new_step_with_validation
    
    print("\n🔧 Using enhanced registration with validation...")
    
    # Register with automatic validation and correction
    warnings = add_new_step_with_validation(
        step_name="MyCustomStep",
        config_class="MyCustomStepConfig",
        builder_name="MyCustomStepStepBuilder", 
        sagemaker_type="Processing",
        description="My custom processing step with validation",
        validation_mode="auto_correct"  # Automatically fixes naming issues
    )
    
    if warnings:
        print("⚠️ Validation warnings:")
        for warning in warnings:
            print(f"   - {warning}")
    else:
        print("✅ Step registered successfully with no issues!")
        
except ImportError:
    print("⚠️ Enhanced registration not available, using basic method")
    
    # Fallback: Register with builder registry
    from cursus.core.base.builder_base import StepBuilderBase
    
    # Create a simple custom builder class
    class MyCustomStepBuilder(StepBuilderBase):
        """Custom step builder for tutorial."""
        
        def __init__(self, config):
            super().__init__(config)
            self.step_name = "my_custom_step"
        
        def build_step(self):
            """Build the custom step."""
            # Implementation would go here
            pass
    
    # Register the builder
    builder_registry = get_global_registry()
    warnings = builder_registry.register_builder("MyCustomStep", MyCustomStepBuilder)
    
    if warnings:
        print("⚠️ Registration warnings:")
        for warning in warnings:
            print(f"   - {warning}")
    else:
        print("✅ Custom builder registered successfully!")
```

## Step 4: Export Registry to JSON (3 minutes)

Let's export the current registry to JSON format for sharing or documentation:

```python
# Export current registry to JSON
current_step_names = STEP_NAMES.copy()

# Save to JSON file
import json
from pathlib import Path

# Create output directory
output_dir = Path("registry_exports")
output_dir.mkdir(exist_ok=True)

# Export to JSON
json_file = output_dir / "step_registry.json"
with open(json_file, 'w') as f:
    json.dump(current_step_names, f, indent=2, sort_keys=True)

print(f"📤 Registry exported to: {json_file}")
print(f"📊 Exported {len(current_step_names)} step definitions")

# Also export derived registries
derived_registries = {
    "config_step_registry": CONFIG_STEP_REGISTRY,
    "builder_step_names": BUILDER_STEP_NAMES,
    "step_names": current_step_names
}

derived_file = output_dir / "derived_registries.json"
with open(derived_file, 'w') as f:
    json.dump(derived_registries, f, indent=2, sort_keys=True)

print(f"📤 Derived registries exported to: {derived_file}")

# Show a sample of the exported data
print(f"\n📋 Sample exported step:")
sample_step = list(current_step_names.items())[0]
print(f"   {sample_step[0]}:")
for key, value in sample_step[1].items():
    print(f"      {key}: {value}")
```

**Expected Output:**
```
📤 Registry exported to: registry_exports/step_registry.json
📊 Exported 17 step definitions
📤 Derived registries exported to: registry_exports/derived_registries.json

📋 Sample exported step:
   Base:
      builder_step_name: StepBuilderBase
      config_class: BasePipelineConfig
      description: Base pipeline configuration
      sagemaker_step_type: Base
      spec_type: Base
```

## Step 5: Workspace Component Discovery (3 minutes)

Let's explore workspace-aware component discovery for multi-developer environments:

```python
# Initialize workspace component registry
from cursus.workspace.core.registry import WorkspaceComponentRegistry

# Set up workspace registry
workspace_registry = WorkspaceComponentRegistry('/path/to/workspace')

# Discover all components across developers
print("🔍 Discovering Workspace Components:")
components = workspace_registry.discover_components()

# Show summary
summary = components['summary']
print(f"   📊 Total components: {summary['total_components']}")
print(f"   👥 Developers: {summary['developers']}")
print(f"   🏗️ Step types: {summary['step_types']}")

# Show component breakdown
print(f"\n📦 Component Breakdown:")
print(f"   Builders: {len(components['builders'])}")
print(f"   Configs: {len(components['configs'])}")
print(f"   Contracts: {len(components['contracts'])}")
print(f"   Specs: {len(components['specs'])}")
print(f"   Scripts: {len(components['scripts'])}")

# Discover components for specific developer
if summary['developers']:
    dev_id = summary['developers'][0]
    print(f"\n👤 Components for {dev_id}:")
    dev_components = workspace_registry.discover_components(dev_id)
    
    for component_type, items in dev_components.items():
        if component_type != 'summary' and items:
            print(f"   {component_type.title()}: {len(items)}")
            for key, info in list(items.items())[:2]:  # Show first 2
                step_name = info.get('step_name', key.split(':')[-1])
                print(f"      - {step_name}")

# Find builder classes with workspace awareness
print(f"\n🏗️ Builder Discovery:")
builder_class = workspace_registry.find_builder_class('processing')
if builder_class:
    print(f"   ✅ Found builder: {builder_class.__name__}")
else:
    print(f"   ❌ No builder found for 'processing'")

# Find builder for specific developer
if summary['developers']:
    dev_builder = workspace_registry.find_builder_class('processing', summary['developers'][0])
    if dev_builder:
        print(f"   ✅ Developer-specific builder: {dev_builder.__name__}")
    else:
        print(f"   ❌ No developer-specific builder found")

# Get workspace summary
workspace_summary = workspace_registry.get_workspace_summary()
print(f"\n📋 Workspace Summary:")
print(f"   Root: {workspace_summary['workspace_root']}")
print(f"   Total: {workspace_summary['total_components']}")
print(f"   Developers: {len(workspace_summary['developers'])}")

# Component counts
counts = workspace_summary['component_counts']
for comp_type, count in counts.items():
    print(f"   {comp_type.title()}: {count}")
```

### 5.1 Component Validation

```python
# Validate component availability for pipeline assembly
print("🔍 Component Validation:")

# Create a mock workspace config for validation
class MockWorkspaceConfig:
    def __init__(self):
        self.steps = []

# Add mock steps
class MockStep:
    def __init__(self, step_name, developer_id):
        self.step_name = step_name
        self.developer_id = developer_id

mock_config = MockWorkspaceConfig()
if summary['developers']:
    # Add some test steps
    mock_config.steps = [
        MockStep('processing', summary['developers'][0]),
        MockStep('training', summary['developers'][0] if len(summary['developers']) > 0 else 'developer_1')
    ]

    # Validate component availability
    validation_result = workspace_registry.validate_component_availability(mock_config)
    
    print(f"   Validation: {'✅ Valid' if validation_result['valid'] else '❌ Invalid'}")
    
    # Show available components
    if validation_result['available_components']:
        print(f"   Available components:")
        for comp in validation_result['available_components']:
            print(f"      ✅ {comp['step_name']} ({comp['developer_id']}) - {comp['component_type']}")
    
    # Show missing components
    if validation_result['missing_components']:
        print(f"   Missing components:")
        for comp in validation_result['missing_components']:
            print(f"      ❌ {comp['step_name']} ({comp['developer_id']}) - {comp['component_type']}")
    
    # Show warnings
    if validation_result['warnings']:
        print(f"   Warnings:")
        for warning in validation_result['warnings']:
            print(f"      ⚠️ {warning}")

# Clear cache for fresh discovery
workspace_registry.clear_cache()
print(f"\n🧹 Component cache cleared")
```

### 5.2 Legacy Workspace Context (Fallback)

```python
# Fallback to legacy workspace context if WorkspaceComponentRegistry not available
try:
    from cursus.registry import (
        set_workspace_context,
        get_workspace_context,
        clear_workspace_context,
        workspace_context,
        list_available_workspaces
    )
    
    print("🏢 Legacy Workspace Context:")
    
    # Check current workspace context
    current_context = get_workspace_context()
    print(f"   Current context: {current_context or 'Core Registry'}")
    
    # List available workspaces
    available_workspaces = list_available_workspaces()
    print(f"   Available workspaces: {len(available_workspaces)}")
    
    # Test context switching with context manager
    if available_workspaces:
        test_workspace = available_workspaces[0]
        with workspace_context(test_workspace):
            context_info = get_registry_info()
            print(f"   Test context: {context_info['workspace_id']}")
            print(f"   Steps in context: {context_info['step_count']}")
    
except ImportError:
    print("⚠️ Legacy workspace context not available")
```

## Step 6: Advanced Registry Operations (2 minutes)

Let's explore some advanced registry operations:

```python
# Advanced registry operations
print("🔧 Advanced Registry Operations:")

# 1. Step name validation and conversion
from cursus.registry import (
    validate_step_name,
    get_canonical_name_from_file_name,
    validate_file_name
)

# Test step name validation
test_names = ["XGBoostTraining", "invalid_name", "MyCustomStep"]
print("\n✅ Step Name Validation:")
for name in test_names:
    is_valid = validate_step_name(name)
    print(f"   {name}: {'✅ Valid' if is_valid else '❌ Invalid'}")

# Test file name to canonical name conversion
test_files = ["xgboost_training", "tabular_preprocessing", "model_evaluation_xgb"]
print("\n🔄 File Name to Canonical Name Conversion:")
for filename in test_files:
    try:
        canonical = get_canonical_name_from_file_name(filename)
        print(f"   {filename} → {canonical}")
    except ValueError as e:
        print(f"   {filename} → ❌ {e}")

# 2. Registry validation
print("\n🔍 Registry Validation:")
builder_registry = get_global_registry()
validation_results = builder_registry.validate_registry()

print(f"   Valid mappings: {len(validation_results['valid'])}")
print(f"   Invalid mappings: {len(validation_results['invalid'])}")
print(f"   Missing builders: {len(validation_results['missing'])}")

if validation_results['invalid']:
    print("   Invalid mappings:")
    for invalid in validation_results['invalid'][:3]:
        print(f"      - {invalid}")

if validation_results['missing']:
    print("   Missing builders:")
    for missing in validation_results['missing'][:3]:
        print(f"      - {missing}")

# 3. Registry statistics
print("\n📊 Detailed Registry Statistics:")
stats = builder_registry.get_registry_stats()
for key, value in stats.items():
    print(f"   {key.replace('_', ' ').title()}: {value}")

# 4. SageMaker step type mapping
from cursus.registry import get_sagemaker_step_type_mapping

sm_mapping = get_sagemaker_step_type_mapping()
print(f"\n🏗️ SageMaker Step Type Distribution:")
for sm_type, steps in sm_mapping.items():
    print(f"   {sm_type}: {len(steps)} steps")
```

## Common Workflows

### Daily Development Workflow

```python
def daily_registry_check():
    """Daily registry health check routine."""
    
    print("🌅 Daily Registry Check")
    
    # 1. Check registry health
    registry_info = get_registry_info()
    print(f"   📊 Total steps: {registry_info['step_count']}")
    print(f"   🏢 Current workspace: {registry_info['workspace_id'] or 'Core'}")
    print(f"   ⚠️ Conflicts: {'Yes' if registry_info['has_conflicts'] else 'No'}")
    
    # 2. Validate builder registry
    builder_registry = get_global_registry()
    validation = builder_registry.validate_registry()
    
    if validation['invalid'] or validation['missing']:
        print("   ❌ Registry issues found:")
        for issue in validation['invalid'][:2]:
            print(f"      - {issue}")
        for missing in validation['missing'][:2]:
            print(f"      - {missing}")
    else:
        print("   ✅ Registry is healthy")
    
    # 3. Check for new steps
    available_steps = get_all_step_names()
    print(f"   🔧 Available step types: {len(available_steps)}")
    
    return len(validation['invalid']) == 0 and len(validation['missing']) == 0

# Run daily check
is_healthy = daily_registry_check()
print(f"\n🏥 Registry Health: {'✅ Healthy' if is_healthy else '⚠️ Needs Attention'}")
```

### Step Discovery Workflow

```python
def discover_steps_by_category(category_filter=None):
    """Discover steps by category or framework."""
    
    print(f"🔍 Step Discovery{f' (filter: {category_filter})' if category_filter else ''}")
    
    # Get all steps
    all_steps = STEP_NAMES
    
    # Filter by category if specified
    if category_filter:
        filtered_steps = {
            name: info for name, info in all_steps.items()
            if category_filter.lower() in info.get('description', '').lower()
            or category_filter.lower() in info.get('sagemaker_step_type', '').lower()
        }
    else:
        filtered_steps = all_steps
    
    # Group by SageMaker type
    from collections import defaultdict
    grouped_steps = defaultdict(list)
    
    for name, info in filtered_steps.items():
        sm_type = info.get('sagemaker_step_type', 'Unknown')
        grouped_steps[sm_type].append((name, info))
    
    # Display results
    for sm_type, steps in sorted(grouped_steps.items()):
        print(f"   📦 {sm_type}: {len(steps)} steps")
        for name, info in steps:
            print(f"      - {name}: {info.get('description', 'No description')}")
    
    return grouped_steps

# Example usage
print("🔍 All Steps:")
all_categories = discover_steps_by_category()

print("\n🔍 Training Steps Only:")
training_steps = discover_steps_by_category("training")

print("\n🔍 Processing Steps Only:")
processing_steps = discover_steps_by_category("processing")
```

### Registry Maintenance Workflow

```python
def maintain_registry():
    """Registry maintenance and cleanup routine."""
    
    print("🔧 Registry Maintenance")
    
    # 1. Validate all registries
    builder_registry = get_global_registry()
    validation = builder_registry.validate_registry()
    
    print(f"   📊 Validation Results:")
    print(f"      Valid: {len(validation['valid'])}")
    print(f"      Invalid: {len(validation['invalid'])}")
    print(f"      Missing: {len(validation['missing'])}")
    
    # 2. Check for naming consistency
    from cursus.registry import validate_step_name
    
    inconsistent_names = []
    for step_name in get_all_step_names():
        if not validate_step_name(step_name):
            inconsistent_names.append(step_name)
    
    if inconsistent_names:
        print(f"   ⚠️ Inconsistent step names: {inconsistent_names}")
    else:
        print(f"   ✅ All step names follow conventions")
    
    # 3. Export current state for backup
    backup_data = {
        "step_names": STEP_NAMES,
        "config_registry": CONFIG_STEP_REGISTRY,
        "builder_names": BUILDER_STEP_NAMES,
        "validation_results": validation,
        "timestamp": str(datetime.now())
    }
    
    from datetime import datetime
    backup_file = f"registry_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(backup_file, 'w') as f:
        json.dump(backup_data, f, indent=2, default=str)
    
    print(f"   💾 Backup saved to: {backup_file}")
    
    return validation

# Run maintenance
maintenance_results = maintain_registry()
```

## Troubleshooting

### Common Issues

**Issue: "Step not found in registry"**
```python
# Debug step resolution
def debug_step_resolution(step_name):
    print(f"🔍 Debugging step resolution for: {step_name}")
    
    # Check if step exists in core registry
    if step_name in STEP_NAMES:
        print(f"   ✅ Found in STEP_NAMES")
        step_info = STEP_NAMES[step_name]
        for key, value in step_info.items():
            print(f"      {key}: {value}")
    else:
        print(f"   ❌ Not found in STEP_NAMES")
        
        # Suggest similar names
        all_names = list(STEP_NAMES.keys())
        similar = [name for name in all_names if step_name.lower() in name.lower()]
        if similar:
            print(f"   💡 Similar names: {similar}")
    
    # Check builder registry
    builder_registry = get_global_registry()
    if builder_registry.is_step_type_supported(step_name):
        print(f"   ✅ Builder available")
    else:
        print(f"   ❌ No builder found")
        supported = builder_registry.list_supported_step_types()
        print(f"   💡 Supported types: {len(supported)} available")

# Example usage
debug_step_resolution("XGBoostTraining")  # Should work
debug_step_resolution("NonExistentStep")  # Should show suggestions
```

**Issue: "Registry validation fails"**
```python
# Fix registry validation issues
def fix_registry_issues():
    print("🔧 Fixing Registry Issues")
    
    builder_registry = get_global_registry()
    validation = builder_registry.validate_registry()
    
    # Show detailed issues
    if validation['invalid']:
        print("   ❌ Invalid mappings:")
        for invalid in validation['invalid']:
            print(f"      - {invalid}")
    
    if validation['missing']:
        print("   ❌ Missing builders:")
        for missing in validation['missing']:
            print(f"      - {missing}")
            
            # Try to suggest fixes
            step_name = missing.split(':')[0]
            if step_name in STEP_NAMES:
                expected_builder = STEP_NAMES[step_name].get('builder_step_name')
                print(f"         Expected builder: {expected_builder}")
    
    # Suggest solutions
    print("\n💡 Suggested fixes:")
    print("   1. Check that all builder files exist")
    print("   2. Verify builder class names match registry")
    print("   3. Ensure all imports are working")
    print("   4. Run builder discovery: builder_registry.discover_builders()")

# Run fix suggestions
fix_registry_issues()
```

**Issue: "Workspace context not working"**
```python
# Debug workspace context
def debug_workspace_context():
    print("🏢 Debugging Workspace Context")
    
    # Check current context
    current = get_workspace_context()
    print(f"   Current context: {current or 'None'}")
    
    # Check available workspaces
    available = list_available_workspaces()
    print(f"   Available workspaces: {len(available)}")
    for workspace in available:
        print(f"      - {workspace}")
    
    # Test context switching
    try:
        with workspace_context("test_workspace"):
            test_context = get_workspace_context()
            print(f"   Test context: {test_context}")
    except Exception as e:
        print(f"   Context switching error: {e}")
    
    # Check hybrid registry availability
    try:
        from cursus.registry.hybrid.manager import UnifiedRegistryManager
        print("   ✅ Hybrid registry available")
    except ImportError:
        print("   ❌ Hybrid registry not available")

# Run workspace debugging
debug_workspace_context()
```

**Issue: "WorkspaceComponentRegistry not discovering components"**
```python
# Debug workspace component discovery
def debug_workspace_component_discovery(workspace_path):
    print(f"🔍 Debugging Workspace Component Discovery: {workspace_path}")
    
    try:
        from cursus.workspace.core.registry import WorkspaceComponentRegistry
        
        # Initialize registry
        workspace_registry = WorkspaceComponentRegistry(workspace_path)
        
        # Check workspace structure
        from pathlib import Path
        workspace_root = Path(workspace_path)
        print(f"   Workspace exists: {workspace_root.exists()}")
        
        if workspace_root.exists():
            # List workspace contents
            contents = list(workspace_root.iterdir())
            print(f"   Workspace contents: {len(contents)} items")
            for item in contents[:5]:  # Show first 5
                print(f"      - {item.name}")
        
        # Try component discovery
        try:
            components = workspace_registry.discover_components()
            summary = components['summary']
            
            print(f"   Discovery results:")
            print(f"      Total components: {summary['total_components']}")
            print(f"      Developers: {summary['developers']}")
            print(f"      Step types: {summary['step_types']}")
            
            # Check for errors
            if 'error' in summary:
                print(f"      ❌ Discovery error: {summary['error']}")
            
        except Exception as e:
            print(f"   ❌ Discovery failed: {e}")
            print(f"   💡 Check workspace structure and permissions")
        
        # Test workspace summary
        try:
            ws_summary = workspace_registry.get_workspace_summary()
            print(f"   Workspace summary available: ✅")
            if 'error' in ws_summary:
                print(f"      ❌ Summary error: {ws_summary['error']}")
        except Exception as e:
            print(f"   ❌ Summary failed: {e}")
        
        # Test cache operations
        try:
            workspace_registry.clear_cache()
            print(f"   Cache operations: ✅")
        except Exception as e:
            print(f"   ❌ Cache operations failed: {e}")
            
    except ImportError:
        print("   ❌ WorkspaceComponentRegistry not available")
        print("   💡 Check if workspace module is installed")

# Example usage
debug_workspace_component_discovery('/path/to/workspace')
```

**Issue: "Component validation fails"**
```python
# Debug component validation
def debug_component_validation():
    print("🔍 Debugging Component Validation")
    
    try:
        from cursus.workspace.core.registry import WorkspaceComponentRegistry
        
        # Initialize with current directory as fallback
        workspace_registry = WorkspaceComponentRegistry('.')
        
        # Create test configuration
        class TestWorkspaceConfig:
            def __init__(self):
                self.steps = []
        
        class TestStep:
            def __init__(self, step_name, developer_id):
                self.step_name = step_name
                self.developer_id = developer_id
        
        # Test with common step names
        test_config = TestWorkspaceConfig()
        test_config.steps = [
            TestStep('processing', 'test_developer'),
            TestStep('training', 'test_developer'),
            TestStep('nonexistent_step', 'test_developer')
        ]
        
        # Run validation
        validation_result = workspace_registry.validate_component_availability(test_config)
        
        print(f"   Validation result:")
        print(f"      Valid: {validation_result['valid']}")
        print(f"      Available: {len(validation_result['available_components'])}")
        print(f"      Missing: {len(validation_result['missing_components'])}")
        print(f"      Warnings: {len(validation_result['warnings'])}")
        
        # Show details
        for comp in validation_result['available_components']:
            print(f"      ✅ {comp['step_name']} - {comp['component_type']}")
        
        for comp in validation_result['missing_components']:
            print(f"      ❌ {comp['step_name']} - {comp['component_type']}")
        
        for warning in validation_result['warnings']:
            print(f"      ⚠️ {warning}")
            
        if 'error' in validation_result:
            print(f"      ❌ Validation error: {validation_result['error']}")
            
    except Exception as e:
        print(f"   ❌ Component validation debugging failed: {e}")

# Run component validation debugging
debug_component_validation()
```

## Next Steps

Congratulations! You've successfully learned the Cursus registry system. Here's what you can do next:

### 1. Create Custom Steps

```python
# Create a complete custom step
from cursus.core.base.builder_base import StepBuilderBase
from cursus.core.base.config_base import BasePipelineConfig

class MyCustomStepConfig(BasePipelineConfig):
    """Configuration for my custom step."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.custom_parameter = kwargs.get('custom_parameter', 'default_value')

class MyCustomStepBuilder(StepBuilderBase):
    """Builder for my custom step."""
    
    def __init__(self, config: MyCustomStepConfig):
        super().__init__(config)
        self.config = config
    
    def build_step(self):
        """Build the custom step."""
        # Your implementation here
        pass

# Register the custom step
builder_registry = get_global_registry()
builder_registry.register_builder("MyCustomStep", MyCustomStepBuilder)
```

### 2. Explore Advanced Features

- **Workspace Development**: Set up isolated development environments
- **Registry Validation**: Implement comprehensive validation rules
- **Step Discovery**: Build automated step discovery systems
- **Registry Integration**: Integrate with CI/CD pipelines

### 3. Integration Examples

```python
# Integration with pipeline compilation
from cursus.core.compiler.dag_compiler import PipelineDAGCompiler

# The registry system automatically resolves steps during compilation
dag_compiler = PipelineDAGCompiler(
    config_path="your_config.json",
    sagemaker_session=your_session,
    role=your_role
)

# Steps are resolved using the registry
pipeline, report = dag_compiler.compile_with_report(your_dag)
```

## Summary

You've successfully learned:

1. ✅ **Registry Initialization**: How to access and explore the registry system
2. ✅ **Step Discovery**: Finding and categorizing existing steps
3. ✅ **Step Registration**: Adding new steps with validation
4. ✅ **JSON Export**: Exporting registry data for sharing
5. ✅ **Workspace Awareness**: Working with workspace contexts
6. ✅ **Advanced Operations**: Validation, maintenance, and troubleshooting

### Key Takeaways

- **Registry is the Single Source of Truth**: All step definitions are centralized
- **Workspace Awareness**: Steps can be context-specific for team collaboration
- **Validation Built-in**: Automatic validation ensures consistency
- **JSON Export**: Easy sharing and documentation of step definitions
- **Builder Integration**: Automatic discovery and registration of step builders

## Additional Resources

### Core Documentation
- **[Registry API Reference](registry_api_reference.md)** - Complete API documentation
- **[Developer Guide](../../0_developer_guide/README.md)** - Comprehensive development guidelines
- **[Step Builder Guide](../../0_developer_guide/step_builder.md)** - Creating custom step builders

### Related Tutorials
- **[Workspace Quick Start](../workspace/workspace_quick_start.md)** - Team collaboration with workspaces
- **[SageMaker Pipeline Quick Start](../main/sagemaker_pipeline_quick_start.md)** - Building complete pipelines
- **[Validation Tutorials](../validation/)** - Step validation and testing

### Advanced Topics
- **[Hybrid Registry Design](../../1_design/workspace_aware_distributed_registry_design.md)** - Architecture details
- **[Registry Standardization](../../1_design/hybrid_registry_standardization_enforcement_design.md)** - Naming conventions and validation
- **[Step Builder Registry Design](../../1_design/step_builder_registry_design.md)** - Builder discovery and registration

Happy step registration! 🚀

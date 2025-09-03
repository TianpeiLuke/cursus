---
tags:
  - code
  - workspace
  - quick_start
  - tutorial
  - getting_started
keywords:
  - workspace tutorial
  - quick start guide
  - developer workspace setup
  - workspace collaboration
  - multi-developer workflow
  - workspace validation
topics:
  - workspace quick start
  - developer onboarding
  - workspace workflow
  - collaborative development
language: python
date of note: 2025-09-02
---

# Workspace Quick Start Guide

## Overview

This 15-minute tutorial will get you up and running with the Cursus workspace system. You'll learn how to create a developer workspace, validate it, discover components, and collaborate with other developers.

## Prerequisites

- Cursus package installed
- Python 3.8+ environment
- Basic familiarity with Python development

## Step 1: Initialize the Workspace API (2 minutes)

First, let's set up the workspace API and verify it's working:

```python
from cursus.workspace import WorkspaceAPI

# Initialize the workspace API
api = WorkspaceAPI()

# Verify the API is working
print("✅ Workspace API initialized successfully")
print(f"Workspace root: {api.base_path}")
```

**Expected Output:**
```
✅ Workspace API initialized successfully
Workspace root: /path/to/your/project/developer_workspaces/developers
```

## Step 2: Create Your First Workspace (3 minutes)

Now let's create a developer workspace with a standard template:

```python
# Create a new developer workspace
result = api.setup_developer_workspace(
    developer_id="your_name",  # Replace with your actual name/ID
    template="ml_pipeline"
)

# Check if the workspace was created successfully
if result.success:
    print(f"🎉 Workspace created successfully!")
    print(f"📁 Location: {result.workspace_path}")
    print(f"👤 Developer ID: {result.developer_id}")
    
    if result.template_used:
        print(f"📋 Template: {result.template_used}")
    
    if result.created_components:
        print(f"🔧 Created components:")
        for component in result.created_components:
            print(f"   - {component}")
else:
    print(f"❌ Workspace creation failed: {result.message}")
    for warning in result.warnings:
        print(f"⚠️  Warning: {warning}")
```

**What this creates:**
- A new directory structure under `developer_workspaces/developers/your_name/`
- Standard ML pipeline template files
- Proper workspace isolation boundaries

## Step 3: Validate Your Workspace (2 minutes)

Let's make sure your workspace is properly configured:

```python
# Validate the workspace
report = api.validate_workspace(result.workspace_path)

print(f"🔍 Workspace Validation Results:")
print(f"Status: {report.status}")

if report.status.name == "HEALTHY":
    print("✅ Workspace is ready for development!")
    print(f"📊 Components found: {sum(report.components.values())}")
    
    # Show component breakdown
    for comp_type, count in report.components.items():
        print(f"   - {comp_type}: {count} files")
else:
    print("⚠️ Workspace has issues:")
    for violation in report.violations:
        print(f"   - {violation}")
```

**Expected Output:**
```
🔍 Workspace Validation Results:
Status: WorkspaceStatus.HEALTHY
✅ Workspace is ready for development!
📊 Components found: 8
   - builders: 2 files
   - configs: 2 files
   - contracts: 2 files
   - specs: 2 files
```

## Step 4: Explore Your Workspace Structure (2 minutes)

Let's examine what was created in your workspace:

```python
# Get detailed workspace information
info = api.get_workspace_info("your_name")

if info:
    print(f"📋 Workspace Details:")
    print(f"   Path: {info.workspace_path}")
    print(f"   Created: {info.created_at}")
    print(f"   Components: {info.component_count}")
    print(f"   Size: {info.size_bytes} bytes")
    print(f"   Valid: {'✅' if info.is_valid else '❌'}")
    
    if info.template_used:
        print(f"   Template: {info.template_used}")
```

**Explore the file structure:**
```python
import os
from pathlib import Path

workspace_path = Path(info.workspace_path)
print(f"\n📁 Workspace Structure:")

# Show the directory tree
for root, dirs, files in os.walk(workspace_path):
    level = root.replace(str(workspace_path), '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    
    subindent = ' ' * 2 * (level + 1)
    for file in files[:3]:  # Show first 3 files per directory
        print(f"{subindent}{file}")
    if len(files) > 3:
        print(f"{subindent}... and {len(files) - 3} more files")
```

## Step 5: Discover Components Across Workspaces (3 minutes)

Now let's see what components are available across all developer workspaces:

```python
# Discover all workspace components
all_components = api.discover_workspace_components()

print(f"🔍 Component Discovery Results:")
print(f"Found {len(all_components)} developer workspaces")

# Show summary statistics
component_stats = {}
for dev_id, dev_components in all_components.items():
    print(f"\n👤 {dev_id}:")
    for comp_type, files in dev_components.items():
        count = len(files)
        print(f"   {comp_type}: {count} files")
        
        # Update global stats
        if comp_type not in component_stats:
            component_stats[comp_type] = 0
        component_stats[comp_type] += count

print(f"\n📊 Total Components Across All Workspaces:")
for comp_type, total_count in component_stats.items():
    print(f"   {comp_type}: {total_count} files")
```

**Discover specific components:**
```python
# Find all XGBoost-related components
print(f"\n🔍 Finding XGBoost Components:")
for dev_id, dev_components in all_components.items():
    builders = dev_components.get('builders', {})
    xgboost_builders = [name for name in builders.keys() if 'xgboost' in name.lower()]
    
    if xgboost_builders:
        print(f"   👤 {dev_id}: {xgboost_builders}")
```

## Step 6: Create a Simple Component (2 minutes)

Let's create a simple step builder in your workspace:

```python
from pathlib import Path

# Create a simple custom step builder
workspace_path = Path(info.workspace_path)
builder_dir = workspace_path / "src" / "cursus_dev" / "steps" / "builders"

# Ensure the directory exists
builder_dir.mkdir(parents=True, exist_ok=True)

# Create a simple builder file
builder_content = '''"""
Custom Data Validation Step Builder
"""
from cursus.steps.builders.base import StepBuilderBase
from cursus.steps.configs.base import BasePipelineConfig
from sagemaker.workflow.steps import ProcessingStep

class CustomDataValidationBuilder(StepBuilderBase):
    """Builder for custom data validation step."""
    
    def __init__(self, config: BasePipelineConfig):
        super().__init__(config)
        self.step_name = "custom_data_validation"
    
    def build_step(self) -> ProcessingStep:
        """Build the data validation processing step."""
        # Implementation would go here
        pass
'''

builder_file = builder_dir / "builder_custom_data_validation_step.py"
builder_file.write_text(builder_content)

print(f"✅ Created custom builder: {builder_file}")

# Create corresponding config file
config_dir = workspace_path / "src" / "cursus_dev" / "steps" / "configs"
config_dir.mkdir(parents=True, exist_ok=True)

config_content = '''"""
Custom Data Validation Step Configuration
"""
from cursus.steps.configs.base import BasePipelineConfig
from pydantic import Field
from typing import List, Optional

class CustomDataValidationConfig(BasePipelineConfig):
    """Configuration for custom data validation step."""
    
    validation_rules: List[str] = Field(
        default_factory=list,
        description="List of validation rules to apply"
    )
    
    fail_on_error: bool = Field(
        default=True,
        description="Whether to fail the pipeline on validation errors"
    )
    
    output_report: bool = Field(
        default=True,
        description="Whether to generate a validation report"
    )
'''

config_file = config_dir / "config_custom_data_validation_step.py"
config_file.write_text(config_content)

print(f"✅ Created custom config: {config_file}")
```

## Step 7: Validate Your New Component (1 minute)

Let's validate that your new component is properly recognized:

```python
# Re-validate the workspace after adding components
report = api.validate_workspace(info.workspace_path)

print(f"🔍 Updated Validation Results:")
print(f"Status: {report.status}")

if report.status.name == "HEALTHY":
    print("✅ Workspace is still healthy with new components!")
    print(f"📊 Components found: {sum(report.components.values())}")
    
    # Show updated component breakdown
    for comp_type, count in report.components.items():
        print(f"   - {comp_type}: {count} files")

# Discover your specific components
your_components = api.discover_workspace_components(developer_id="your_name")
print(f"\n🔧 Your Components:")
for comp_type, files in your_components.get("your_name", {}).items():
    print(f"   {comp_type}:")
    for filename in files.keys():
        print(f"      - {filename}")
```

## Step 8: Workspace Collaboration Example (Optional)

If you have multiple developers, here's how to collaborate:

```python
# Example: Building a pipeline using components from multiple developers
from cursus.workspace.core import WorkspaceStepDefinition

# Define a multi-developer pipeline
pipeline_steps = {
    "data_loading": WorkspaceStepDefinition(
        developer_id="data_engineer_alice",
        step_name="data_loading",
        step_type="CradleDataLoading",
        config_data={
            "dataset": "customer_data",
            "format": "parquet",
            "bucket": "ml-data-bucket"
        }
    ),
    "data_validation": WorkspaceStepDefinition(
        developer_id="your_name",  # Using your custom component!
        step_name="data_validation", 
        step_type="CustomDataValidation",
        config_data={
            "validation_rules": ["check_nulls", "check_schema"],
            "fail_on_error": True,
            "output_report": True
        }
    ),
    "model_training": WorkspaceStepDefinition(
        developer_id="ml_engineer_bob",
        step_name="model_training",
        step_type="XGBoostTraining",
        config_data={
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100
        }
    )
}

print("🤝 Multi-Developer Pipeline Configuration:")
for step_name, step_def in pipeline_steps.items():
    print(f"   {step_name}: {step_def.step_type} (by {step_def.developer_id})")
```

## Common Workflows

### Daily Development Workflow

```python
# 1. Check workspace health
def daily_workspace_check(developer_id: str):
    """Daily workspace health check routine."""
    
    print(f"🌅 Daily Workspace Check for {developer_id}")
    
    # Get workspace info
    info = api.get_workspace_info(developer_id)
    if not info:
        print("❌ Workspace not found!")
        return False
    
    # Validate workspace
    report = api.validate_workspace(info.workspace_path)
    
    if report.status.name == "HEALTHY":
        print("✅ Workspace is healthy")
        print(f"📊 {info.component_count} components available")
        return True
    else:
        print("⚠️ Workspace issues detected:")
        for violation in report.violations:
            print(f"   - {violation}")
        return False

# Run daily check
daily_workspace_check("your_name")
```

### Component Promotion Workflow

```python
# 2. Promote a component to staging
def promote_component_to_staging(developer_id: str, component_name: str):
    """Promote a workspace component to integration staging."""
    
    print(f"🚀 Promoting {component_name} from {developer_id} to staging")
    
    try:
        result = api.promote_workspace_component(
            developer_id=developer_id,
            component_path=f"src/cursus_dev/steps/builders/builder_{component_name}_step.py",
            target="staging"
        )
        
        if result.success:
            print(f"✅ Component promoted successfully!")
            print(f"📁 Staged at: {result.target_path}")
        else:
            print(f"❌ Promotion failed: {result.message}")
            
    except Exception as e:
        print(f"❌ Promotion error: {e}")

# Example promotion (uncomment to try)
# promote_component_to_staging("your_name", "custom_data_validation")
```

### Workspace Cleanup

```python
# 3. Clean up workspace
def cleanup_workspace_routine(developer_id: str):
    """Regular workspace cleanup routine."""
    
    print(f"🧹 Cleaning up workspace for {developer_id}")
    
    result = api.cleanup_workspace(developer_id, deep_clean=False)
    
    if result.success:
        print(f"✅ Cleanup completed!")
        print(f"🗑️  Removed {result.files_removed} files")
        print(f"💾 Freed {result.space_freed} bytes")
    else:
        print(f"❌ Cleanup failed: {result.message}")

# Run cleanup (uncomment to try)
# cleanup_workspace_routine("your_name")
```

## Troubleshooting

### Issue: "Workspace not found"
```python
# Check if workspace directory exists
import os
workspace_path = f"developer_workspaces/developers/your_name"
if not os.path.exists(workspace_path):
    print("❌ Workspace directory doesn't exist")
    print("💡 Try running setup_developer_workspace() again")
```

### Issue: "Component discovery returns empty"
```python
# Debug component discovery
components = api.discover_workspace_components(developer_id="your_name")
if not components:
    print("❌ No components found")
    print("💡 Check file naming conventions:")
    print("   - Builders: builder_<type>_step.py")
    print("   - Configs: config_<type>_step.py")
```

### Issue: "Validation fails"
```python
# Get detailed validation info
report = api.validate_workspace("developer_workspaces/developers/your_name")
if report.violations:
    print("🔍 Validation Issues:")
    for violation in report.violations:
        print(f"   - {violation}")
    print("💡 Check workspace structure and file permissions")
```

## Next Steps

Congratulations! You've successfully:

1. ✅ Created your first developer workspace
2. ✅ Validated workspace health
3. ✅ Discovered components across workspaces
4. ✅ Created a custom component
5. ✅ Learned collaboration workflows

### What's Next?

1. **Explore Advanced Features**: Check out the [Workspace API Reference](workspace_api_reference.md) for advanced usage patterns

2. **Build Real Components**: Start developing actual step builders, configs, and scripts for your ML pipelines

3. **Collaborate**: Work with other developers to build multi-workspace pipelines

4. **Integrate with CI/CD**: Set up automated workspace validation in your development pipeline

5. **Monitor Workspace Health**: Implement regular health checks and cleanup routines

### Additional Resources

- **[Workspace API Reference](workspace_api_reference.md)** - Complete API documentation
- **[Developer Guide](../0_developer_guide/README.md)** - Comprehensive development guidelines
- **[Design Documents](../1_design/)** - Architectural details and design decisions

## Summary

You now have a fully functional developer workspace and understand the basic workflow for:

- Creating and managing workspaces
- Validating workspace health
- Discovering and sharing components
- Collaborating with other developers
- Promoting components through the integration pipeline

The workspace system enables isolated development while maintaining the ability to collaborate and share components across your team. Happy coding! 🚀

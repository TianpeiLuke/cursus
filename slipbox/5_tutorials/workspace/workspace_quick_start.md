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
  - workspace API
  - component discovery
  - workspace validation
  - simplified architecture
topics:
  - workspace quick start
  - developer onboarding
  - workspace workflow
  - component management
language: python
date of note: 2025-09-29
---

# Workspace Quick Start Guide

## Overview

This 15-minute tutorial will get you up and running with the simplified Cursus workspace system. You'll learn how to use the unified WorkspaceAPI to discover components, validate workspaces, and work with workspace-aware pipelines using the proven step catalog architecture.

## Prerequisites

- Cursus package installed
- Python 3.8+ environment
- Basic familiarity with Python development

## Step 1: Initialize the Workspace API (2 minutes)

The new workspace system provides a single, unified API built on the step catalog's proven architecture.

```python
from cursus.workspace import WorkspaceAPI
from pathlib import Path

# Package-only mode (discovers only core package components)
api = WorkspaceAPI()
print("✅ Package-only workspace API initialized")

# Single workspace directory
api = WorkspaceAPI(workspace_dirs=Path("/projects/alpha"))
print("✅ Single workspace API initialized")

# Multiple workspace directories with flexible organization
api = WorkspaceAPI(workspace_dirs=[
    Path("/teams/data_science/experiments"),
    Path("/projects/beta/custom_steps"),
    Path("/features/recommendation/components")
])
print("✅ Multi-workspace API initialized")

# Get workspace summary
summary = api.get_workspace_summary()
print(f"📊 Workspace Configuration:")
print(f"   Total workspaces: {summary['total_workspaces']}")
print(f"   Workspace directories: {summary['workspace_directories']}")
```

**Expected Output:**
```
✅ Multi-workspace API initialized
📊 Workspace Configuration:
   Total workspaces: 3
   Workspace directories: ['/teams/data_science/experiments', '/projects/beta/custom_steps', '/features/recommendation/components']
```

## Step 2: Discover Components Across Workspaces (3 minutes)

The workspace API uses the step catalog's proven discovery mechanism to find components across all configured workspaces.

```python
# Discover all components across all workspaces
all_components = api.discover_components()
print(f"🔍 Total components discovered: {len(all_components)}")

# Discover components from specific workspace (using directory name as workspace ID)
if api.workspace_dirs:
    workspace_id = api.workspace_dirs[0].name  # Use first workspace as example
    workspace_components = api.discover_components(workspace_id=workspace_id)
    print(f"📁 Components in '{workspace_id}' workspace: {len(workspace_components)}")
    
    # Show first few components
    for component in workspace_components[:3]:
        print(f"   - {component}")

# Discover core package components
core_components = api.discover_components(workspace_id="core")
print(f"📦 Core package components: {len(core_components)}")

# Get cross-workspace component breakdown
cross_workspace = api.get_cross_workspace_components()
print(f"\n📊 Components by workspace:")
for workspace_id, components in cross_workspace.items():
    print(f"   {workspace_id}: {len(components)} components")
```

**Expected Output:**
```
🔍 Total components discovered: 25
📁 Components in 'experiments' workspace: 8
   - custom_data_loader
   - feature_transformer
   - model_validator
📦 Core package components: 17

📊 Components by workspace:
   core: 17 components
   experiments: 8 components
   custom_steps: 5 components
   components: 3 components
```

## Step 3: Get Detailed Component Information (2 minutes)

Use the step catalog integration to get detailed information about discovered components.

```python
# Get detailed information about a specific component
component_name = "xgboost_training"  # Use an actual component from your discovery
info = api.get_component_info(component_name)

if info:
    print(f"📋 Component Details for '{component_name}':")
    print(f"   Name: {info.step_name}")
    print(f"   Workspace: {info.workspace_id}")
    print(f"   Available files:")
    
    for comp_type, metadata in info.file_components.items():
        if metadata:
            print(f"      {comp_type}: {metadata.path}")
else:
    print(f"❌ Component '{component_name}' not found")

# Find specific component files
builder_path = api.find_component_file(component_name, "builder")
config_path = api.find_component_file(component_name, "config")
script_path = api.find_component_file(component_name, "script")

print(f"\n🔍 Component File Locations:")
if builder_path:
    print(f"   Builder: {builder_path}")
if config_path:
    print(f"   Config: {config_path}")
if script_path:
    print(f"   Script: {script_path}")
```

**Expected Output:**
```
📋 Component Details for 'xgboost_training':
   Name: xgboost_training
   Workspace: core
   Available files:
      builder: /path/to/cursus/steps/builders/builder_xgboost_training_step.py
      config: /path/to/cursus/steps/configs/config_xgboost_training_step.py
      script: /path/to/cursus/steps/scripts/xgboost_training.py

🔍 Component File Locations:
   Builder: /path/to/cursus/steps/builders/builder_xgboost_training_step.py
   Config: /path/to/cursus/steps/configs/config_xgboost_training_step.py
   Script: /path/to/cursus/steps/scripts/xgboost_training.py
```

## Step 4: Search and Filter Components (2 minutes)

Use the step catalog's search capabilities to find components with fuzzy matching.

```python
# Search for components with fuzzy matching
search_query = "xgboost"
search_results = api.search_components(search_query)
print(f"🔍 Search results for '{search_query}': {len(search_results)} found")

for result in search_results[:3]:  # Show first 3 results
    print(f"   - {result.step_name} (workspace: {result.workspace_id})")

# Search within specific workspace
if api.workspace_dirs:
    workspace_id = api.workspace_dirs[0].name
    workspace_search = api.search_components(search_query, workspace_id=workspace_id)
    print(f"🔍 '{search_query}' in '{workspace_id}' workspace: {len(workspace_search)} found")

# Check component availability
component_exists = api.is_component_available("xgboost_training")
print(f"✅ XGBoost training available: {component_exists}")

# Check availability in specific workspace
workspace_exists = api.is_component_available("custom_preprocessing", workspace_id="experiments")
print(f"✅ Custom preprocessing in experiments: {workspace_exists}")
```

**Expected Output:**
```
🔍 Search results for 'xgboost': 3 found
   - xgboost_training (workspace: core)
   - xgboost_model_eval (workspace: core)
   - custom_xgboost_tuner (workspace: experiments)
🔍 'xgboost' in 'experiments' workspace: 1 found
✅ XGBoost training available: True
✅ Custom preprocessing in experiments: False
```

## Step 5: Validate Workspace Structure and Components (3 minutes)

The workspace system provides flexible validation that works with any directory organization.

```python
# Validate workspace directory structure (flexible validation)
if api.workspace_dirs:
    for workspace_dir in api.workspace_dirs:
        validation = api.validate_workspace_structure(workspace_dir)
        
        print(f"🔍 Workspace Validation: {workspace_dir.name}")
        print(f"   Valid: {'✅' if validation['valid'] else '❌'}")
        print(f"   Exists: {validation['exists']}")
        print(f"   Readable: {validation['readable']}")
        print(f"   Components found: {validation['components_found']}")
        
        if validation['warnings']:
            print(f"   Warnings:")
            for warning in validation['warnings']:
                print(f"      - {warning}")

# Validate components in a specific workspace
workspace_id = "core"  # Validate core components
validation_result = api.validate_workspace_components(workspace_id)

print(f"\n📊 Component Validation Results for '{workspace_id}':")
print(f"   Valid: {'✅' if validation_result.is_valid else '❌'}")
print(f"   Components validated: {validation_result.details.get('validated_components', 0)}")
print(f"   Total components: {validation_result.details.get('total_components', 0)}")

if validation_result.errors:
    print(f"   Errors:")
    for error in validation_result.errors:
        print(f"      - {error}")

if validation_result.warnings:
    print(f"   Warnings:")
    for warning in validation_result.warnings:
        print(f"      - {warning}")
```

**Expected Output:**
```
🔍 Workspace Validation: experiments
   Valid: ✅
   Exists: True
   Readable: True
   Components found: 8

📊 Component Validation Results for 'core':
   Valid: ✅
   Components validated: 17
   Total components: 17
```

## Step 6: Validate Component Quality (2 minutes)

Assess the quality of individual components using the integrated validation framework.

```python
# Validate quality of specific components
components_to_check = ["xgboost_training", "dummy_training", "package"]

print("🔍 Component Quality Assessment:")
for component in components_to_check:
    if api.is_component_available(component):
        quality_result = api.validate_component_quality(component)
        
        score = quality_result.details.get('quality_score', 0)
        completeness = quality_result.details.get('component_completeness', 0)
        missing = quality_result.details.get('missing_components', [])
        
        print(f"\n   {component}:")
        print(f"      Valid: {'✅' if quality_result.is_valid else '❌'}")
        print(f"      Quality score: {score}/100")
        print(f"      Component completeness: {completeness}")
        
        if missing:
            print(f"      Missing components: {', '.join(missing)}")
        
        if quality_result.errors:
            print(f"      Errors:")
            for error in quality_result.errors:
                print(f"         - {error}")
    else:
        print(f"\n   {component}: ❌ Not available")
```

**Expected Output:**
```
🔍 Component Quality Assessment:

   xgboost_training:
      Valid: ✅
      Quality score: 85/100
      Component completeness: 4

   dummy_training:
      Valid: ✅
      Quality score: 70/100
      Component completeness: 3
      Missing components: contract, spec

   package:
      Valid: ✅
      Quality score: 90/100
      Component completeness: 5
```

## Step 7: Cross-Workspace Compatibility (2 minutes)

Check compatibility between components from different workspaces.

```python
# Check cross-workspace compatibility
workspaces = api.list_all_workspaces()
print(f"🔍 Available workspaces: {workspaces}")

if len(workspaces) > 1:
    # Check compatibility between first two workspaces
    compatibility = api.validate_cross_workspace_compatibility(workspaces[:2])
    
    print(f"\n🤝 Cross-Workspace Compatibility:")
    print(f"   Compatible: {'✅' if compatibility.is_compatible else '❌'}")
    
    if compatibility.issues:
        print(f"   Issues:")
        for issue in compatibility.issues:
            print(f"      - {issue}")
    
    print(f"   Compatibility matrix:")
    for workspace_id, info in compatibility.compatibility_matrix.items():
        conflicts = info.get('conflicts', 0)
        total = info.get('total_components', 0)
        print(f"      {workspace_id}: {total} components, {conflicts} conflicts")
else:
    print("⚠️ Need multiple workspaces for compatibility testing")
```

**Expected Output:**
```
🔍 Available workspaces: ['core', 'experiments', 'custom_steps']

🤝 Cross-Workspace Compatibility:
   Compatible: ✅
   Compatibility matrix:
      core: 17 components, 0 conflicts
      experiments: 8 components, 0 conflicts
```

## Step 8: Create Workspace-Aware Pipeline (2 minutes)

Create pipelines that can use components from multiple workspaces.

```python
from cursus.api.dag.base_dag import PipelineDAG

# Create a simple pipeline DAG
dag = PipelineDAG()
dag.add_node("data_loading")
dag.add_node("preprocessing")
dag.add_node("training")
dag.add_edge("data_loading", "preprocessing")
dag.add_edge("preprocessing", "training")

print("📋 Pipeline DAG created:")
print(f"   Nodes: {list(dag.nodes)}")
print(f"   Edges: {list(dag.edges)}")

# Create workspace-aware pipeline (requires config file)
# Note: This would typically use an actual config file
config_path = "config/example_pipeline.json"

try:
    pipeline = api.create_workspace_pipeline(dag, config_path)
    
    if pipeline:
        print(f"✅ Workspace-aware pipeline created successfully")
        print(f"   Pipeline uses components from configured workspaces")
        print(f"   Step catalog automatically resolves component locations")
    else:
        print("❌ Pipeline creation failed (config file may not exist)")
        
except Exception as e:
    print(f"⚠️ Pipeline creation skipped: {e}")
    print("💡 This requires a valid pipeline configuration file")
```

**Expected Output:**
```
📋 Pipeline DAG created:
   Nodes: ['data_loading', 'preprocessing', 'training']
   Edges: [('data_loading', 'preprocessing'), ('preprocessing', 'training')]
⚠️ Pipeline creation skipped: [Errno 2] No such file or directory: 'config/example_pipeline.json'
💡 This requires a valid pipeline configuration file
```

## Step 9: System Status and Monitoring (1 minute)

Monitor the workspace system's performance and status.

```python
# Get comprehensive system status
status = api.get_system_status()

print("📊 System Status:")
print(f"   API success rate: {status['workspace_api']['success_rate']:.2%}")
print(f"   Total API calls: {status['workspace_api']['metrics']['api_calls']}")
print(f"   Successful operations: {status['workspace_api']['metrics']['successful_operations']}")
print(f"   Failed operations: {status['workspace_api']['metrics']['failed_operations']}")

print(f"\n📁 Workspace Manager:")
print(f"   Total components: {status['manager']['total_components']}")
print(f"   Total workspaces: {status['manager']['total_workspaces']}")

print(f"\n🔍 Validator:")
validator_metrics = status['validator']['metrics']
print(f"   Validations performed: {validator_metrics['validations_performed']}")
print(f"   Components validated: {validator_metrics['components_validated']}")

# Refresh catalog if needed
print(f"\n🔄 Catalog Management:")
refresh_success = api.refresh_catalog()
print(f"   Catalog refresh: {'✅ Success' if refresh_success else '❌ Failed'}")

# Get updated component count after refresh
updated_components = api.discover_components()
print(f"   Components after refresh: {len(updated_components)}")
```

**Expected Output:**
```
📊 System Status:
   API success rate: 95.24%
   Total API calls: 21
   Successful operations: 20
   Failed operations: 1

📁 Workspace Manager:
   Total components: 25
   Total workspaces: 3

🔍 Validator:
   Validations performed: 3
   Components validated: 25

🔄 Catalog Management:
   Catalog refresh: ✅ Success
   Components after refresh: 25
```

## Common Workflows

### Daily Development Workflow

```python
def daily_workspace_check(api: WorkspaceAPI):
    """Daily workspace health check routine."""
    
    print("🌅 Daily Workspace Health Check")
    
    # Get workspace summary
    summary = api.get_workspace_summary()
    print(f"📊 Workspace Overview:")
    print(f"   Total workspaces: {summary['total_workspaces']}")
    print(f"   Total components: {summary['total_components']}")
    
    # Validate each workspace
    healthy_workspaces = 0
    for workspace_dir in api.workspace_dirs:
        validation = api.validate_workspace_structure(workspace_dir)
        if validation['valid']:
            healthy_workspaces += 1
            print(f"   ✅ {workspace_dir.name}: {validation['components_found']} components")
        else:
            print(f"   ❌ {workspace_dir.name}: Issues detected")
            for warning in validation.get('warnings', []):
                print(f"      - {warning}")
    
    # Overall health assessment
    health_rate = healthy_workspaces / len(api.workspace_dirs) if api.workspace_dirs else 1.0
    print(f"\n🏥 Overall Health: {health_rate:.1%} ({healthy_workspaces}/{len(api.workspace_dirs)} workspaces healthy)")
    
    return health_rate >= 0.8

# Run daily check
health_status = daily_workspace_check(api)
print(f"Daily check result: {'✅ Healthy' if health_status else '⚠️ Needs attention'}")
```

### Component Discovery and Analysis

```python
def analyze_workspace_components(api: WorkspaceAPI):
    """Analyze components across all workspaces."""
    
    cross_workspace = api.get_cross_workspace_components()
    
    print("🔍 Workspace Component Analysis:")
    
    # Component distribution
    total_components = sum(len(components) for components in cross_workspace.values())
    print(f"   Total components: {total_components}")
    
    # Workspace breakdown
    for workspace_id, components in cross_workspace.items():
        percentage = (len(components) / total_components * 100) if total_components > 0 else 0
        print(f"   {workspace_id}: {len(components)} components ({percentage:.1f}%)")
    
    # Component type analysis
    component_types = {'builder': 0, 'config': 0, 'script': 0, 'contract': 0, 'spec': 0}
    
    for workspace_id, components in cross_workspace.items():
        for component in components[:5]:  # Sample first 5 components per workspace
            info = api.get_component_info(component)
            if info:
                for comp_type in component_types.keys():
                    if info.file_components.get(comp_type):
                        component_types[comp_type] += 1
    
    print(f"\n📋 Component Types (sampled):")
    for comp_type, count in component_types.items():
        print(f"   {comp_type}: {count}")
    
    return cross_workspace

# Run component analysis
component_analysis = analyze_workspace_components(api)
```

### Quality Monitoring Dashboard

```python
def quality_monitoring_dashboard(api: WorkspaceAPI):
    """Generate a quality monitoring dashboard."""
    
    print("📊 Component Quality Dashboard")
    print("=" * 50)
    
    # Get all components
    all_components = api.discover_components()
    
    # Quality categories
    quality_categories = {
        'high': [],      # 80-100
        'medium': [],    # 60-79
        'low': [],       # 40-59
        'critical': []   # 0-39
    }
    
    validation_errors = 0
    
    # Assess quality for each component
    for component in all_components[:10]:  # Limit to first 10 for demo
        try:
            quality_result = api.validate_component_quality(component)
            
            if quality_result.is_valid:
                score = quality_result.details.get('quality_score', 0)
                workspace_id = quality_result.details.get('workspace_id', 'unknown')
                
                if score >= 80:
                    quality_categories['high'].append((component, score, workspace_id))
                elif score >= 60:
                    quality_categories['medium'].append((component, score, workspace_id))
                elif score >= 40:
                    quality_categories['low'].append((component, score, workspace_id))
                else:
                    quality_categories['critical'].append((component, score, workspace_id))
            else:
                validation_errors += 1
                quality_categories['critical'].append((component, 0, 'error'))
                
        except Exception as e:
            validation_errors += 1
            quality_categories['critical'].append((component, 0, 'exception'))
    
    # Display dashboard
    total_assessed = sum(len(components) for components in quality_categories.values())
    
    print(f"Components assessed: {total_assessed}")
    print(f"Validation errors: {validation_errors}")
    print()
    
    for category, components in quality_categories.items():
        if components:
            percentage = len(components) / total_assessed * 100 if total_assessed > 0 else 0
            print(f"{category.upper()} QUALITY ({percentage:.1f}%):")
            
            for component, score, workspace_id in components:
                if score > 0:
                    print(f"   {component}: {score}/100 ({workspace_id})")
                else:
                    print(f"   {component}: Validation failed ({workspace_id})")
            print()
    
    return quality_categories

# Run quality dashboard
quality_dashboard = quality_monitoring_dashboard(api)
```

## Advanced Usage Patterns

### Multi-Workspace Pipeline Development

```python
# Advanced multi-workspace setup
api = WorkspaceAPI(workspace_dirs=[
    Path("/teams/data_engineering/etl_components"),
    Path("/teams/ml_engineering/model_components"),
    Path("/teams/feature_engineering/feature_components"),
    Path("/projects/production/validated_components")
])

# Discover specialized components by workspace
workspaces = {
    'etl': api.discover_components(workspace_id="etl_components"),
    'models': api.discover_components(workspace_id="model_components"),
    'features': api.discover_components(workspace_id="feature_components"),
    'production': api.discover_components(workspace_id="validated_components")
}

print("🏗️ Multi-Team Component Inventory:")
for team, components in workspaces.items():
    print(f"   {team}: {len(components)} components")
    
    # Show component types
    component_types = {}
    for component in components[:3]:  # Sample first 3
        info = api.get_component_info(component)
        if info:
            available_types = [t for t in info.file_components.keys() if info.file_components[t]]
            for comp_type in available_types:
                component_types[comp_type] = component_types.get(comp_type, 0) + 1
    
    if component_types:
        type_summary = ', '.join([f"{t}:{c}" for t, c in component_types.items()])
        print(f"      Types: {type_summary}")

# Cross-team compatibility check
all_workspace_ids = list(workspaces.keys())
if len(all_workspace_ids) > 1:
    compatibility = api.validate_cross_workspace_compatibility(all_workspace_ids)
    print(f"\n🤝 Cross-Team Compatibility: {'✅' if compatibility.is_compatible else '❌'}")
    
    if not compatibility.is_compatible:
        print("   Issues to resolve:")
        for issue in compatibility.issues:
            print(f"      - {issue}")
```

### Component Promotion Workflow

```python
def component_promotion_workflow(api: WorkspaceAPI, component_name: str, source_workspace: str):
    """Demonstrate component promotion workflow."""
    
    print(f"🚀 Component Promotion Workflow: {component_name}")
    print("=" * 50)
    
    # Step 1: Validate component exists and quality
    if not api.is_component_available(component_name, workspace_id=source_workspace):
        print(f"❌ Component '{component_name}' not found in workspace '{source_workspace}'")
        return False
    
    quality_result = api.validate_component_quality(component_name)
    quality_score = quality_result.details.get('quality_score', 0)
    
    print(f"📊 Component Quality Assessment:")
    print(f"   Quality score: {quality_score}/100")
    print(f"   Valid: {'✅' if quality_result.is_valid else '❌'}")
    
    if quality_score < 70:
        print(f"⚠️ Quality score too low for promotion (minimum: 70)")
        return False
    
    # Step 2: Dry run promotion
    print(f"\n🧪 Dry Run Promotion:")
    try:
        dry_result = api.promote_component_to_core(
            step_name=component_name,
            source_workspace_id=source_workspace,
            dry_run=True
        )
        
        print(f"   Dry run result: {'✅ Success' if dry_result.success else '❌ Failed'}")
        print(f"   Message: {dry_result.message}")
        
        if not dry_result.success:
            print(f"   Cannot proceed with promotion")
            return False
            
    except Exception as e:
        print(f"   ❌ Dry run failed: {e}")
        return False
    
    # Step 3: Actual promotion (commented out for safety)
    print(f"\n🎯 Ready for Promotion:")
    print(f"   Component: {component_name}")
    print(f"   Source workspace: {source_workspace}")
    print(f"   Quality score: {quality_score}/100")
    print(f"   Dry run: ✅ Passed")
    print(f"   💡 Uncomment promotion code to execute")
    
    # Uncomment the following lines to perform actual promotion:
    # promotion_result = api.promote_component_to_core(
    #     step_name=component_name,
    #     source_workspace_id=source_workspace,
    #     dry_run=False
    # )
    # 
    # if promotion_result.success:
    #     print(f"✅ Component promoted successfully!")
    #     print(f"Details: {promotion_result.details}")
    # else:
    #     print(f"❌ Promotion failed: {promotion_result.message}")
    
    return True

# Example promotion workflow (using a component that exists)
if api.workspace_dirs:
    # Try to promote a component from the first workspace
    workspace_id = api.workspace_dirs[0].name
    workspace_components = api.discover_components(workspace_id=workspace_id)
    
    if workspace_components:
        example_component = workspace_components[0]
        promotion_success = component_promotion_workflow(api, example_component, workspace_id)
        print(f"Promotion workflow result: {'✅ Ready' if promotion_success else '❌ Not ready'}")
```

## Troubleshooting

### Common Issues and Solutions

**Issue: No components discovered**
```python
# Diagnostic steps
print("🔍 Troubleshooting: No components discovered")

# Check workspace configuration
summary = api.get_workspace_summary()
print(f"1. Workspace configuration:")
print(f"   Directories: {summary['workspace_directories']}")
print(f"   Total workspaces: {summary['total_workspaces']}")

# Validate each workspace directory
print(f"\n2. Workspace validation:")
for workspace_dir in api.workspace_dirs:
    validation = api.validate_workspace_structure(workspace_dir)
    print(f"   {workspace_dir}:")
    print(f"      Exists: {validation['exists']}")
    print(f"      Readable: {validation['readable']}")
    print(f"      Components: {validation['components_found']}")

# Check catalog refresh
print(f"\n3. Catalog refresh:")
refresh_success = api.refresh_catalog()
print(f"   Refresh result: {'✅ Success' if refresh_success else '❌ Failed'}")

# Re-check components after refresh
updated_components = api.discover_components()
print(f"   Components after refresh: {len(updated_components)}")
```

**Issue: Component validation fails**
```python
# Detailed component diagnostics
def diagnose_component(api: WorkspaceAPI, component_name: str):
    print(f"🔍 Diagnosing component: {component_name}")
    
    # Check availability
    available = api.is_component_available(component_name)
    print(f"   Available: {'✅' if available else '❌'}")
    
    if not available:
        print(f"   💡 Component not found in any workspace")
        return
    
    # Get component info
    info = api.get_component_info(component_name)
    if info:
        print(f"   Workspace: {info.workspace_id}")
        print(f"   File components:")
        
        for comp_type, metadata in info.file_components.items():
            if metadata:
                file_path = Path(metadata.path)
                exists = file_path.exists()
                readable = file_path.is_file() if exists else False
                
                print(f"      {comp_type}: {metadata.path}")
                print(f"         Exists: {'✅' if exists else '❌'}")
                print(f"         Readable: {'✅' if readable else '❌'}")
    
    # Quality assessment
    quality_result = api.validate_component_quality(component_name)
    print(f"   Quality valid: {'✅' if quality_result.is_valid else '❌'}")
    print(f"   Quality score: {quality_result.details.get('quality_score', 0)}/100")
    
    if quality_result.errors:
        print(f"   Errors:")
        for error in quality_result.errors:
            print(f"      - {error}")

# Example diagnosis
components = api.discover_components()
if components:
    diagnose_component(api, components[0])
```

**Issue: Cross-workspace compatibility problems**
```python
# Detailed compatibility analysis
def analyze_compatibility_issues(api: WorkspaceAPI):
    print("🔍 Cross-Workspace Compatibility Analysis")
    
    workspaces = api.list_all_workspaces()
    print(f"Available workspaces: {workspaces}")
    
    if len(workspaces) < 2:
        print("⚠️ Need at least 2 workspaces for compatibility analysis")
        return
    
    compatibility = api.validate_cross_workspace_compatibility(workspaces)
    
    print(f"\nCompatibility result: {'✅ Compatible' if compatibility.is_compatible else '❌ Issues foun

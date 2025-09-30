---
tags:
  - code
  - workspace
  - api_reference
  - documentation
  - implementation
keywords:
  - workspace API
  - step catalog integration
  - workspace management
  - component discovery
  - workspace validation
  - simplified architecture
topics:
  - workspace API reference
  - workspace management
  - component discovery
  - workspace validation
language: python
date of note: 2025-09-29
---

# Workspace API Reference

## Overview

The Workspace API provides a unified, simplified interface for workspace-aware operations built on the proven step catalog architecture. This dramatically simplified system eliminates 84% of code redundancy while preserving all original functionality through direct step catalog integration.

## Key Architecture Benefits

- **84% Code Reduction**: From 4,200 → 620 lines of code
- **Flexible Organization**: No hardcoded workspace structure requirements
- **Deployment Agnostic**: Works across all deployment scenarios
- **Proven Integration**: Built on step catalog's proven dual search space architecture

## Core API Classes

### WorkspaceAPI

The unified entry point for all workspace operations.

```python
from cursus.workspace import WorkspaceAPI

# Package-only mode (discovers only core package components)
api = WorkspaceAPI()

# Single workspace directory
api = WorkspaceAPI(workspace_dirs=Path("/projects/alpha"))

# Multiple workspace directories with flexible organization
api = WorkspaceAPI(workspace_dirs=[
    Path("/teams/data_science/experiments"),
    Path("/projects/beta/custom_steps"),
    Path("/features/recommendation/components")
])
```

## Component Discovery and Management

### discover_components()

Discover components using step catalog's proven discovery mechanism.

**Signature:**
```python
def discover_components(self, workspace_id: Optional[str] = None) -> List[str]
```

**Parameters:**
- `workspace_id`: Optional workspace filter (uses directory name as ID)

**Returns:** List of discovered component names

**Example:**
```python
api = WorkspaceAPI(workspace_dirs=[
    Path("/projects/alpha"),
    Path("/projects/beta")
])

# Discover all components across all workspaces
all_components = api.discover_components()
print(f"Total components: {len(all_components)}")

# Discover components from specific workspace
alpha_components = api.discover_components(workspace_id="alpha")
print(f"Alpha workspace components: {alpha_components}")

# Discover core package components only
core_components = api.discover_components(workspace_id="core")
print(f"Core package components: {core_components}")
```

### get_component_info()

Get detailed component information using step catalog.

**Signature:**
```python
def get_component_info(self, step_name: str) -> Optional[StepInfo]
```

**Parameters:**
- `step_name`: Name of the component

**Returns:** `StepInfo` object with component details, or None if not found

**Example:**
```python
# Get component information
info = api.get_component_info("xgboost_training")

if info:
    print(f"Component: {info.step_name}")
    print(f"Workspace: {info.workspace_id}")
    print(f"Available files:")
    for comp_type, metadata in info.file_components.items():
        if metadata:
            print(f"  {comp_type}: {metadata.path}")
else:
    print("Component not found")
```

### find_component_file()

Find specific component file using step catalog.

**Signature:**
```python
def find_component_file(self, step_name: str, component_type: str) -> Optional[Path]
```

**Parameters:**
- `step_name`: Name of the step
- `component_type`: Type of component ('builder', 'config', 'contract', 'spec', 'script')

**Returns:** Path to component file, or None if not found

**Example:**
```python
# Find specific component files
builder_path = api.find_component_file("xgboost_training", "builder")
config_path = api.find_component_file("xgboost_training", "config")
script_path = api.find_component_file("xgboost_training", "script")

if builder_path:
    print(f"Builder found: {builder_path}")
if config_path:
    print(f"Config found: {config_path}")
if script_path:
    print(f"Script found: {script_path}")
```

### search_components()

Search components with fuzzy matching using step catalog.

**Signature:**
```python
def search_components(self, query: str, workspace_id: Optional[str] = None) -> List[Any]
```

**Parameters:**
- `query`: Search query string
- `workspace_id`: Optional workspace filter

**Returns:** List of search results sorted by relevance

**Example:**
```python
# Search for XGBoost-related components
xgboost_results = api.search_components("xgboost")
print(f"Found {len(xgboost_results)} XGBoost components")

# Search within specific workspace
alpha_xgboost = api.search_components("xgboost", workspace_id="alpha")
print(f"XGBoost components in alpha workspace: {len(alpha_xgboost)}")
```

## Workspace Management

### get_workspace_summary()

Get comprehensive workspace summary.

**Signature:**
```python
def get_workspace_summary(self) -> Dict[str, Any]
```

**Returns:** Dictionary with workspace configuration and component information

**Example:**
```python
summary = api.get_workspace_summary()

print(f"Workspace Configuration:")
print(f"  Total workspaces: {summary['total_workspaces']}")
print(f"  Total components: {summary['total_components']}")
print(f"  Workspace directories: {summary['workspace_directories']}")

print(f"\nComponents by workspace:")
for workspace_id, count in summary['workspace_components'].items():
    print(f"  {workspace_id}: {count} components")

print(f"\nAPI Metrics:")
for metric, value in summary['api_metrics'].items():
    print(f"  {metric}: {value}")
```

### validate_workspace_structure()

Validate workspace directory structure (flexible validation).

**Signature:**
```python
def validate_workspace_structure(self, workspace_dir: Path) -> Dict[str, Any]
```

**Parameters:**
- `workspace_dir`: Workspace directory to validate

**Returns:** Dictionary with validation results

**Example:**
```python
workspace_path = Path("/projects/alpha")
validation = api.validate_workspace_structure(workspace_path)

print(f"Workspace Validation:")
print(f"  Valid: {validation['valid']}")
print(f"  Exists: {validation['exists']}")
print(f"  Readable: {validation['readable']}")
print(f"  Components found: {validation['components_found']}")

if validation['warnings']:
    print(f"  Warnings:")
    for warning in validation['warnings']:
        print(f"    - {warning}")
```

### get_cross_workspace_components()

Get components organized by workspace.

**Signature:**
```python
def get_cross_workspace_components(self) -> Dict[str, List[str]]
```

**Returns:** Dictionary mapping workspace IDs to component lists

**Example:**
```python
cross_workspace = api.get_cross_workspace_components()

print("Components by workspace:")
for workspace_id, components in cross_workspace.items():
    print(f"\n{workspace_id} ({len(components)} components):")
    for component in components[:5]:  # Show first 5
        print(f"  - {component}")
    if len(components) > 5:
        print(f"  ... and {len(components) - 5} more")
```

## Pipeline Creation

### create_workspace_pipeline()

Create pipeline using workspace-aware components.

**Signature:**
```python
def create_workspace_pipeline(self, dag: PipelineDAG, config_path: str) -> Optional[Any]
```

**Parameters:**
- `dag`: Pipeline DAG definition
- `config_path`: Path to pipeline configuration

**Returns:** Generated pipeline object, or None if creation fails

**Example:**
```python
from cursus.api.dag.base_dag import PipelineDAG

# Create pipeline DAG
dag = PipelineDAG()
dag.add_node("data_loading")
dag.add_node("preprocessing")
dag.add_node("training")
dag.add_edge("data_loading", "preprocessing")
dag.add_edge("preprocessing", "training")

# Create workspace-aware pipeline
pipeline = api.create_workspace_pipeline(dag, "config/pipeline_config.json")

if pipeline:
    print(f"✅ Pipeline created successfully")
    print(f"Pipeline steps: {len(dag.nodes)}")
else:
    print("❌ Pipeline creation failed")
```

## Validation and Quality Assessment

### validate_workspace_components()

Validate all components in a workspace.

**Signature:**
```python
def validate_workspace_components(self, workspace_id: str) -> ValidationResult
```

**Parameters:**
- `workspace_id`: ID of the workspace to validate

**Returns:** `ValidationResult` with validation details

**Example:**
```python
# Validate workspace components
result = api.validate_workspace_components("alpha")

print(f"Workspace Validation Results:")
print(f"  Valid: {result.is_valid}")
print(f"  Components validated: {result.details.get('validated_components', 0)}")
print(f"  Total components: {result.details.get('total_components', 0)}")

if result.errors:
    print(f"  Errors:")
    for error in result.errors:
        print(f"    - {error}")

if result.warnings:
    print(f"  Warnings:")
    for warning in result.warnings:
        print(f"    - {warning}")
```

### validate_component_quality()

Validate quality of a specific component.

**Signature:**
```python
def validate_component_quality(self, step_name: str) -> ValidationResult
```

**Parameters:**
- `step_name`: Name of the component to validate

**Returns:** `ValidationResult` with quality assessment

**Example:**
```python
# Validate component quality
quality_result = api.validate_component_quality("xgboost_training")

print(f"Component Quality Assessment:")
print(f"  Valid: {quality_result.is_valid}")
print(f"  Quality score: {quality_result.details.get('quality_score', 0)}/100")
print(f"  Component completeness: {quality_result.details.get('component_completeness', 0)}")

missing = quality_result.details.get('missing_components', [])
if missing:
    print(f"  Missing components: {', '.join(missing)}")
```

### validate_cross_workspace_compatibility()

Validate compatibility between workspace components.

**Signature:**
```python
def validate_cross_workspace_compatibility(self, workspace_ids: List[str]) -> CompatibilityResult
```

**Parameters:**
- `workspace_ids`: List of workspace IDs to check compatibility

**Returns:** `CompatibilityResult` with compatibility analysis

**Example:**
```python
# Check compatibility between workspaces
compatibility = api.validate_cross_workspace_compatibility(["alpha", "beta"])

print(f"Cross-Workspace Compatibility:")
print(f"  Compatible: {compatibility.is_compatible}")

if compatibility.issues:
    print(f"  Issues:")
    for issue in compatibility.issues:
        print(f"    - {issue}")

print(f"  Compatibility matrix:")
for workspace_id, info in compatibility.compatibility_matrix.items():
    print(f"    {workspace_id}: {info['total_components']} components, {info['conflicts']} conflicts")
```

## Component Integration and Promotion

### promote_component_to_core()

Promote workspace component to core package.

**Signature:**
```python
def promote_component_to_core(self, step_name: str, source_workspace_id: str, 
                             dry_run: bool = True) -> IntegrationResult
```

**Parameters:**
- `step_name`: Name of the component to promote
- `source_workspace_id`: ID of the source workspace
- `dry_run`: If True, only validate promotion without executing

**Returns:** `IntegrationResult` with promotion details

**Example:**
```python
# Dry run promotion (validation only)
dry_result = api.promote_component_to_core(
    step_name="custom_preprocessing",
    source_workspace_id="alpha",
    dry_run=True
)

print(f"Promotion Dry Run:")
print(f"  Success: {dry_result.success}")
print(f"  Message: {dry_result.message}")

if dry_result.success:
    # Actual promotion
    promotion_result = api.promote_component_to_core(
        step_name="custom_preprocessing",
        source_workspace_id="alpha",
        dry_run=False
    )
    
    if promotion_result.success:
        print(f"✅ Component promoted successfully")
        print(f"Details: {promotion_result.details}")
    else:
        print(f"❌ Promotion failed: {promotion_result.message}")
```

### integrate_cross_workspace_components()

Integrate components from multiple workspaces.

**Signature:**
```python
def integrate_cross_workspace_components(self, target_workspace_id: str, 
                                       source_components: List[Dict[str, str]]) -> IntegrationResult
```

**Parameters:**
- `target_workspace_id`: ID of the target workspace
- `source_components`: List of dicts with 'step_name' and 'source_workspace_id'

**Returns:** `IntegrationResult` with integration details

**Example:**
```python
# Define components to integrate
source_components = [
    {"step_name": "data_loader", "source_workspace_id": "alpha"},
    {"step_name": "feature_engineer", "source_workspace_id": "beta"},
    {"step_name": "model_trainer", "source_workspace_id": "gamma"}
]

# Integrate components
integration_result = api.integrate_cross_workspace_components(
    target_workspace_id="production",
    source_components=source_components
)

print(f"Cross-Workspace Integration:")
print(f"  Success: {integration_result.success}")
print(f"  Message: {integration_result.message}")

if integration_result.details:
    print(f"  Details: {integration_result.details}")
```

### rollback_promotion()

Rollback component promotion from core package.

**Signature:**
```python
def rollback_promotion(self, step_name: str) -> IntegrationResult
```

**Parameters:**
- `step_name`: Name of the component to rollback

**Returns:** `IntegrationResult` with rollback details

**Example:**
```python
# Rollback a promoted component
rollback_result = api.rollback_promotion("custom_preprocessing")

print(f"Promotion Rollback:")
print(f"  Success: {rollback_result.success}")
print(f"  Message: {rollback_result.message}")

if rollback_result.success:
    print(f"✅ Component rollback completed")
else:
    print(f"❌ Rollback failed")
```

## System Maintenance

### refresh_catalog()

Refresh the step catalog to pick up new components.

**Signature:**
```python
def refresh_catalog(self) -> bool
```

**Returns:** True if refresh successful, False otherwise

**Example:**
```python
# Refresh catalog after adding new components
success = api.refresh_catalog()

if success:
    print("✅ Catalog refreshed successfully")
    
    # Verify new components are discovered
    updated_components = api.discover_components()
    print(f"Total components after refresh: {len(updated_components)}")
else:
    print("❌ Catalog refresh failed")
```

### get_system_status()

Get comprehensive system status and metrics.

**Signature:**
```python
def get_system_status(self) -> Dict[str, Any]
```

**Returns:** Dictionary with system status and metrics from all components

**Example:**
```python
status = api.get_system_status()

print("System Status:")
print(f"  Success rate: {status['workspace_api']['success_rate']:.2%}")
print(f"  Total API calls: {status['workspace_api']['metrics']['api_calls']}")
print(f"  Successful operations: {status['workspace_api']['metrics']['successful_operations']}")
print(f"  Failed operations: {status['workspace_api']['metrics']['failed_operations']}")

print(f"\nWorkspace Manager:")
print(f"  Total components: {status['manager']['total_components']}")
print(f"  Total workspaces: {status['manager']['total_workspaces']}")

print(f"\nValidator:")
print(f"  Validations performed: {status['validator']['metrics']['validations_performed']}")
print(f"  Components validated: {status['validator']['metrics']['components_validated']}")

print(f"\nIntegrator:")
print(f"  Promotions: {status['integrator']['metrics'].get('promotions', 0)}")
print(f"  Integrations: {status['integrator']['metrics'].get('integrations', 0)}")
```

## Convenience Methods

### list_all_workspaces()

List all available workspace IDs.

**Signature:**
```python
def list_all_workspaces(self) -> List[str]
```

**Returns:** List of workspace IDs

**Example:**
```python
workspaces = api.list_all_workspaces()
print(f"Available workspaces: {workspaces}")
```

### get_workspace_component_count()

Get count of components in a specific workspace.

**Signature:**
```python
def get_workspace_component_count(self, workspace_id: str) -> int
```

**Parameters:**
- `workspace_id`: ID of the workspace

**Returns:** Number of components in the workspace

**Example:**
```python
alpha_count = api.get_workspace_component_count("alpha")
beta_count = api.get_workspace_component_count("beta")
core_count = api.get_workspace_component_count("core")

print(f"Component counts:")
print(f"  Alpha workspace: {alpha_count}")
print(f"  Beta workspace: {beta_count}")
print(f"  Core package: {core_count}")
```

### is_component_available()

Check if a component is available in the specified workspace.

**Signature:**
```python
def is_component_available(self, step_name: str, workspace_id: Optional[str] = None) -> bool
```

**Parameters:**
- `step_name`: Name of the component
- `workspace_id`: Optional workspace filter

**Returns:** True if component is available, False otherwise

**Example:**
```python
# Check component availability
if api.is_component_available("xgboost_training"):
    print("✅ XGBoost training component is available")
else:
    print("❌ XGBoost training component not found")

# Check in specific workspace
if api.is_component_available("custom_preprocessing", workspace_id="alpha"):
    print("✅ Custom preprocessing available in alpha workspace")
else:
    print("❌ Custom preprocessing not found in alpha workspace")
```

## Data Models

### ValidationResult

Result of validation operations.

```python
class ValidationResult:
    def __init__(self, is_valid: bool, errors: Optional[List[str]] = None, 
                 warnings: Optional[List[str]] = None, details: Optional[Dict[str, Any]] = None):
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.details = details or {}
```

### CompatibilityResult

Result of compatibility validation.

```python
class CompatibilityResult:
    def __init__(self, is_compatible: bool, issues: Optional[List[str]] = None,
                 compatibility_matrix: Optional[Dict[str, Dict[str, bool]]] = None):
        self.is_compatible = is_compatible
        self.issues = issues or []
        self.compatibility_matrix = compatibility_matrix or {}
```

### IntegrationResult

Result of integration operations.

```python
class IntegrationResult:
    def __init__(self, success: bool, message: str, details: Optional[Dict[str, Any]] = None):
        self.success = success
        self.message = message
        self.details = details or {}
```

## Error Handling

### Best Practices

```python
from cursus.workspace import WorkspaceAPI

api = WorkspaceAPI(workspace_dirs=[Path("/projects/alpha")])

try:
    # Component discovery
    components = api.discover_components()
    
    if not components:
        print("⚠️ No components found")
        # Check workspace configuration
        summary = api.get_workspace_summary()
        print(f"Workspace directories: {summary['workspace_directories']}")
    
    # Component validation
    for component in components[:3]:  # Validate first 3 components
        quality_result = api.validate_component_quality(component)
        
        if quality_result.is_valid:
            print(f"✅ {component}: Quality score {quality_result.details.get('quality_score', 0)}/100")
        else:
            print(f"❌ {component}: Quality issues found")
            for error in quality_result.errors:
                print(f"    - {error}")
    
    # Cross-workspace compatibility
    workspaces = api.list_all_workspaces()
    if len(workspaces) > 1:
        compatibility = api.validate_cross_workspace_compatibility(workspaces)
        
        if compatibility.is_compatible:
            print("✅ All workspaces are compatible")
        else:
            print("⚠️ Compatibility issues found:")
            for issue in compatibility.issues:
                print(f"    - {issue}")

except Exception as e:
    print(f"❌ Workspace operation failed: {e}")
    
    # Get system status for debugging
    status = api.get_system_status()
    if 'error' in status:
        print(f"System error: {status['error']}")
    else:
        print(f"API success rate: {status['workspace_api']['success_rate']:.2%}")
```

## Advanced Usage Patterns

### Multi-Workspace Pipeline Development

```python
# Set up multi-workspace environment
api = WorkspaceAPI(workspace_dirs=[
    Path("/teams/data_engineering/components"),
    Path("/teams/ml_engineering/models"),
    Path("/teams/feature_engineering/transformers")
])

# Discover components across all workspaces
all_components = api.get_cross_workspace_components()

print("Multi-Workspace Component Inventory:")
for workspace_id, components in all_components.items():
    print(f"\n{workspace_id} ({len(components)} components):")
    
    # Categorize components by type
    component_types = {}
    for component in components:
        info = api.get_component_info(component)
        if info:
            # Infer component type from available files
            if info.file_components.get('builder'):
                comp_type = 'pipeline_step'
            elif info.file_components.get('script'):
                comp_type = 'script'
            else:
                comp_type = 'other'
            
            if comp_type not in component_types:
                component_types[comp_type] = []
            component_types[comp_type].append(component)
    
    for comp_type, comps in component_types.items():
        print(f"  {comp_type}: {len(comps)} components")

# Validate cross-workspace compatibility
workspace_ids = list(all_components.keys())
compatibility = api.validate_cross_workspace_compatibility(workspace_ids)

if compatibility.is_compatible:
    print("\n✅ All workspaces are compatible for collaboration")
else:
    print("\n⚠️ Compatibility issues detected:")
    for issue in compatibility.issues:
        print(f"  - {issue}")
```

### Component Quality Monitoring

```python
def monitor_component_quality(api: WorkspaceAPI):
    """Monitor component quality across all workspaces."""
    
    all_components = api.discover_components()
    quality_report = {
        'total_components': len(all_components),
        'high_quality': 0,
        'medium_quality': 0,
        'low_quality': 0,
        'failed_validation': 0,
        'component_details': {}
    }
    
    for component in all_components:
        try:
            quality_result = api.validate_component_quality(component)
            
            if quality_result.is_valid:
                score = quality_result.details.get('quality_score', 0)
                
                if score >= 80:
                    quality_report['high_quality'] += 1
                    quality_level = 'high'
                elif score >= 60:
                    quality_report['medium_quality'] += 1
                    quality_level = 'medium'
                else:
                    quality_report['low_quality'] += 1
                    quality_level = 'low'
                
                quality_report['component_details'][component] = {
                    'quality_level': quality_level,
                    'score': score,
                    'workspace_id': quality_result.details.get('workspace_id', 'unknown')
                }
            else:
                quality_report['failed_validation'] += 1
                quality_report['component_details'][component] = {
                    'quality_level': 'failed',
                    'errors': quality_result.errors
                }
                
        except Exception as e:
            quality_report['failed_validation'] += 1
            quality_report['component_details'][component] = {
                'quality_level': 'error',
                'error': str(e)
            }
    
    return quality_report

# Run quality monitoring
quality_report = monitor_component_quality(api)

print("Component Quality Report:")
print(f"  Total components: {quality_report['total_components']}")
print(f"  High quality (80-100): {quality_report['high_quality']}")
print(f"  Medium quality (60-79): {quality_report['medium_quality']}")
print(f"  Low quality (0-59): {quality_report['low_quality']}")
print(f"  Failed validation: {quality_report['failed_validation']}")

# Show low quality components that need attention
low_quality_components = [
    name for name, details in quality_report['component_details'].items()
    if details['quality_level'] in ['low', 'failed', 'error']
]

if low_quality_components:
    print(f"\n⚠️ Components needing attention ({len(low_quality_components)}):")
    for component in low_quality_components[:5]:  # Show first 5
        details = quality_report['component_details'][component]
        print(f"  - {component}: {details['quality_level']}")
```

## Integration with Existing Systems

### Step Catalog Integration

The workspace API is built directly on the step catalog, providing seamless integration:

```python
# Access underlying step catalog
catalog = api.catalog

# Use step catalog methods directly
step_info = catalog.get_step_info("xgboost_training")
search_results = catalog.search_steps("preprocessing")

# Get step catalog metrics
if hasattr(catalog, 'get_metrics_report'):
    catalog_metrics = catalog.get_metrics_report()
    print(f"Step catalog metrics: {catalog_metrics}")
```

### Pipeline Assembly Integration

```python
from cursus.api.dag.base_dag import PipelineDAG

# Create pipeline using workspace components
dag = PipelineDAG()
dag.add_node("data_loading")
dag.add_node("preprocessing") 
dag.add_node("training")
dag.add_edge("data_loading", "preprocessing")
dag.add_edge("preprocessing", "training")

# Create workspace-aware pipeline
pipeline = api.create_workspace_pipeline(dag, "config/multi_workspace_pipeline.json")

if pipeline:
    print("✅ Multi-workspace pipeline created successfully")
    
    # The pipeline automatically uses components from configured workspaces
    # based on the step catalog's workspace-aware discovery
```

## Troubleshooting

### Common Issues and Solutions

**Issue: No components discovered**
```python
# Check workspace configuration
summary = api.get_workspace_summary()
print(f"Configured workspaces: {summary['workspace_directories']}")

# Validate workspace structure
for workspace_dir in api.workspace_dirs:
    validation = api.validate_workspace_structure(workspace_dir)
    print(f"{workspace_dir}: valid={validation['valid']}, components={validation['components_found']}")

# Refresh catalog
success = api.refresh_catalog()
print(f"Catalog refresh: {'success' if success else 'failed'}")
```

**Issue: Component validation fails**
```python
# Get detailed component information
component_name = "problematic_component"
info = api.get_component_info(component_name)

if info:
    print(f"Component info for {component_name}:")
    print(f"  Workspace: {info.workspace_id}")
    print(f"  Available files:")
    for comp_type, metadata in info.file_components.items():
        if metadata:
            file_path = Path(metadata.path)
            exists = file_path.exists()
            print(f"    {comp_type}: {metadata.path} (exists: {exists})")
else:
    print(f"Component {component_name} not found in step catalog")
```

**Issue: Cross-workspace compatibility problems**
```python
# Detailed compatibility analysis
workspaces = api.list_all_workspaces()
compatibility = api.validate_cross_workspace_compatibility(workspaces)

print("Detailed Compatibility Analysis:")
for workspace_id, info in compatibility.compatibility_matrix.items():
    print(f"\n{workspace_id}:")
    print(f"  Total components: {info['total_components']}")
    print(f"  Conflicts: {info['conflicts']}")
    
    if info['conflicting_components']:
        print(f"  Conflicting components:")
        for conflict in info['conflicting_components']:
            print(f"    - {conflict}")
```

## Performance Considerations

### Optimization Tips

1. **Use workspace_id filters** when possible to reduce discovery scope
2. **Cache component information** for frequently accessed components
3. **Batch validation operations** instead of individual calls
4. **Refresh catalog only when necessary** (after adding new components)

### Performance Monitoring

```python
# Monitor API performance
status = api.get_system_status()
api_metrics = status['workspace_api']['metrics']

print("API Performance Metrics:")
print(f"  Total API calls: {api_metrics['api_calls']}")
print(f"  Success rate: {status['workspace_api']['success_rate']:.2%}")
print(f"  Average operations per call: {api_metrics['successful_operations'] / max(1, api_metrics['api_calls']):.2f}")
```

## API Reference Summary

| Method | Purpose | Returns |
|--------|---------|---------|
| `discover_components()` | Find components using step catalog | `List[str]` |
| `get_component_info()` | Get component details | `Optional[StepInfo]` |
| `find_component_file()` | Find specific component file | `Optional[Path]` |
| `search_components()` | Search with fuzzy matching | `List[Any]` |
| `get_workspace_summary()` | Get workspace overview | `Dict[str, Any]` |
| `validate_workspace_structure()` | Validate workspace directory | `Dict[str, Any]` |
| `get_cross_workspace_components()` | Get components by workspace | `Dict[str, List[str]]` |
| `create_workspace_pipeline()` | Create workspace-aware pipeline | `Optional[Any]` |
| `validate_workspace_components()` | Validate workspace components | `ValidationResult` |
| `validate_component_quality()` | Validate component quality | `ValidationResult` |
| `validate_cross_workspace_compatibility()` | Check compatibility | `CompatibilityResult` |
| `promote_component_to_core()` | Promote component | `IntegrationResult` |
| `integrate_cross_workspace_components()` | Integrate components | `IntegrationResult` |
| `rollback_promotion()` | Rollback promotion | `IntegrationResult` |
| `refresh_catalog()` | Refresh

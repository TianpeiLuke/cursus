# Integration Patterns Examples

This document provides examples of integration patterns for the workspace-aware validation system.

## Basic Integration Patterns

### End-to-End Workspace Setup and Validation

```python
from cursus.validation.workspace.workspace_manager import WorkspaceManager
from cursus.validation.workspace.workspace_orchestrator import WorkspaceValidationOrchestrator
from cursus.validation.workspace.models import WorkspaceConfig, WorkspaceSettings

def complete_workspace_setup_and_validation(workspace_root, developer_id):
    """Complete workflow from workspace creation to validation"""
    
    # Step 1: Initialize workspace manager
    manager = WorkspaceManager(workspace_root)
    
    # Step 2: Create workspace structure
    workspace_path = manager.create_workspace_structure(developer_id)
    print(f"Created workspace structure at: {workspace_path}")
    
    # Step 3: Create and save configuration
    settings = WorkspaceSettings(
        python_version="3.9",
        dependencies=["numpy", "pandas", "scikit-learn"],
        environment_variables={"MODEL_PATH": "/models"},
        validation_rules={"strict_typing": True}
    )
    
    config = WorkspaceConfig(
        workspace_name=f"Workspace for {developer_id}",
        developer_id=developer_id,
        version="1.0.0",
        description="Development workspace",
        workspace_settings=settings
    )
    
    manager.save_workspace_config(developer_id, config)
    print("Configuration saved")
    
    # Step 4: Validate workspace
    orchestrator = WorkspaceValidationOrchestrator(workspace_root)
    
    # Discover components automatically
    workspace_info = manager.get_workspace_info(developer_id)
    available_components = workspace_info.get('builders', [])
    
    if available_components:
        # Run validation
        result = orchestrator.validate_workspace(
            developer_id=developer_id,
            components=available_components,
            validation_types=["alignment", "builder"],
            levels=[1, 2, 3, 4]
        )
        
        print(f"Validation result: {'PASSED' if result.overall_passed else 'FAILED'}")
        return result
    else:
        print("No components found for validation")
        return None

# Usage
result = complete_workspace_setup_and_validation("/path/to/workspaces", "developer_1")
```

### Multi-Component Validation Pipeline

```python
def multi_component_validation_pipeline(workspace_root, developer_id, components):
    """Validate multiple components with different strategies"""
    
    orchestrator = WorkspaceValidationOrchestrator(workspace_root)
    
    # Phase 1: Individual component validation
    print("Phase 1: Individual component validation")
    individual_results = {}
    
    for component in components:
        result = orchestrator.validate_workspace(
            developer_id=developer_id,
            components=[component],
            validation_types=["alignment", "builder"],
            levels=[1, 2, 3, 4]
        )
        individual_results[component] = result.overall_passed
        print(f"  {component}: {'PASS' if result.overall_passed else 'FAIL'}")
    
    # Phase 2: Batch validation of passing components
    passing_components = [c for c, passed in individual_results.items() if passed]
    
    if len(passing_components) > 1:
        print(f"\nPhase 2: Batch validation of {len(passing_components)} components")
        batch_result = orchestrator.validate_workspace(
            developer_id=developer_id,
            components=passing_components,
            validation_types=["alignment", "builder"],
            levels=[1, 2, 3, 4]
        )
        print(f"  Batch validation: {'PASS' if batch_result.overall_passed else 'FAIL'}")
        return batch_result
    else:
        print("Not enough passing components for batch validation")
        return None

# Usage
components = ["xgboost_trainer", "data_processor", "model_evaluator"]
batch_result = multi_component_validation_pipeline(
    "/path/to/workspaces", 
    "developer_1", 
    components
)
```

## Cross-Workspace Integration

### Multi-Developer Collaboration Validation

```python
def cross_developer_validation(workspace_root, developers, shared_components):
    """Validate shared components across multiple developers"""
    
    orchestrator = WorkspaceValidationOrchestrator(workspace_root)
    
    # Validate each developer's implementation of shared components
    developer_results = {}
    
    for developer_id in developers:
        print(f"\nValidating {developer_id}'s implementations...")
        
        # Check which shared components this developer has
        manager = WorkspaceManager(workspace_root)
        workspace_info = manager.get_workspace_info(developer_id)
        available_builders = workspace_info.get('builders', [])
        
        # Find intersection with shared components
        developer_shared = [c for c in shared_components if c in available_builders]
        
        if developer_shared:
            result = orchestrator.validate_workspace(
                developer_id=developer_id,
                components=developer_shared,
                validation_types=["alignment", "builder"],
                levels=[1, 2, 3, 4]
            )
            
            developer_results[developer_id] = {
                'components': developer_shared,
                'result': result,
                'passed': result.overall_passed
            }
            
            print(f"  {developer_id}: {len(developer_shared)} components, {'PASS' if result.overall_passed else 'FAIL'}")
        else:
            print(f"  {developer_id}: No shared components found")
    
    # Generate compatibility report
    print(f"\nCompatibility Report:")
    for component in shared_components:
        implementations = []
        for dev_id, dev_result in developer_results.items():
            if component in dev_result['components']:
                status = 'PASS' if dev_result['passed'] else 'FAIL'
                implementations.append(f"{dev_id}:{status}")
        
        if implementations:
            print(f"  {component}: {', '.join(implementations)}")
        else:
            print(f"  {component}: No implementations found")
    
    return developer_results

# Usage
developers = ["developer_1", "developer_2", "developer_3"]
shared_components = ["data_loader", "feature_processor", "model_trainer"]

compatibility_results = cross_developer_validation(
    "/path/to/workspaces", 
    developers, 
    shared_components
)
```

### Workspace Synchronization Pattern

```python
def workspace_synchronization_pattern(workspace_root, source_developer, target_developers):
    """Synchronize workspace configurations and validate consistency"""
    
    manager = WorkspaceManager(workspace_root)
    orchestrator = WorkspaceValidationOrchestrator(workspace_root)
    
    # Step 1: Load source configuration
    source_config = manager.load_workspace_config(source_developer)
    print(f"Loaded configuration from {source_developer}")
    
    # Step 2: Validate source workspace first
    source_result = orchestrator.validate_workspace(
        developer_id=source_developer,
        validation_types=["alignment", "builder"],
        levels=[1, 2, 3, 4]
    )
    
    if not source_result.overall_passed:
        print(f"Source workspace {source_developer} validation failed")
        return False
    
    print(f"Source workspace {source_developer} validation passed")
    
    # Step 3: Synchronize to target workspaces
    sync_results = {}
    
    for target_developer in target_developers:
        print(f"\nSynchronizing to {target_developer}...")
        
        try:
            # Create target workspace if it doesn't exist
            if not manager.workspace_exists(target_developer):
                manager.create_workspace_structure(target_developer)
            
            # Copy configuration with modifications
            target_config = source_config.copy(deep=True)
            target_config.developer_id = target_developer
            target_config.workspace_name = f"Synced workspace for {target_developer}"
            
            # Save target configuration
            manager.save_workspace_config(target_developer, target_config)
            
            # Validate target workspace
            target_result = orchestrator.validate_workspace(
                developer_id=target_developer,
                validation_types=["alignment", "builder"],
                levels=[1, 2, 3, 4]
            )
            
            sync_results[target_developer] = {
                'synced': True,
                'validated': target_result.overall_passed,
                'result': target_result
            }
            
            status = "SUCCESS" if target_result.overall_passed else "VALIDATION_FAILED"
            print(f"  {target_developer}: {status}")
            
        except Exception as e:
            sync_results[target_developer] = {
                'synced': False,
                'error': str(e)
            }
            print(f"  {target_developer}: SYNC_FAILED - {e}")
    
    return sync_results

# Usage
sync_results = workspace_synchronization_pattern(
    "/path/to/workspaces",
    "developer_1",  # source
    ["developer_2", "developer_3", "developer_4"]  # targets
)
```

## Advanced Integration Patterns

### Hierarchical Validation Pattern

```python
def hierarchical_validation_pattern(workspace_root, developer_id):
    """Validate workspace using hierarchical approach"""
    
    orchestrator = WorkspaceValidationOrchestrator(workspace_root)
    
    # Define validation hierarchy
    validation_hierarchy = [
        {
            'name': 'Level 1 - Basic Structure',
            'validation_types': ['alignment'],
            'levels': [1],
            'required': True
        },
        {
            'name': 'Level 2 - Contract Alignment',
            'validation_types': ['alignment'],
            'levels': [2],
            'required': True
        },
        {
            'name': 'Level 3 - Dependency Validation',
            'validation_types': ['alignment'],
            'levels': [3],
            'required': False
        },
        {
            'name': 'Level 4 - Configuration Alignment',
            'validation_types': ['alignment'],
            'levels': [4],
            'required': False
        },
        {
            'name': 'Builder Validation',
            'validation_types': ['builder'],
            'levels': None,
            'required': True
        },
        {
            'name': 'Full Integration',
            'validation_types': ['alignment', 'builder'],
            'levels': [1, 2, 3, 4],
            'required': False
        }
    ]
    
    results = {}
    overall_success = True
    
    for phase in validation_hierarchy:
        print(f"\nRunning {phase['name']}...")
        
        # Prepare validation parameters
        validation_params = {
            'developer_id': developer_id,
            'validation_types': phase['validation_types']
        }
        
        if phase['levels']:
            validation_params['levels'] = phase['levels']
        
        # Run validation
        result = orchestrator.validate_workspace(**validation_params)
        results[phase['name']] = result
        
        success = result.overall_passed
        status = "PASS" if success else "FAIL"
        print(f"  {phase['name']}: {status}")
        
        # Check if this phase is required
        if phase['required'] and not success:
            print(f"  Required phase failed: {phase['name']}")
            overall_success = False
            
            # Stop at first required failure
            break
        elif not success:
            print(f"  Optional phase failed: {phase['name']}")
    
    print(f"\nHierarchical validation: {'PASSED' if overall_success else 'FAILED'}")
    return overall_success, results

# Usage
success, phase_results = hierarchical_validation_pattern(
    "/path/to/workspaces", 
    "developer_1"
)
```

### Dependency-Aware Validation Pattern

```python
def dependency_aware_validation_pattern(workspace_root, developer_id, component_dependencies):
    """Validate components in dependency order"""
    
    orchestrator = WorkspaceValidationOrchestrator(workspace_root)
    
    # Topological sort of components based on dependencies
    def topological_sort(dependencies):
        from collections import defaultdict, deque
        
        # Build graph
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        all_nodes = set()
        
        for component, deps in dependencies.items():
            all_nodes.add(component)
            for dep in deps:
                all_nodes.add(dep)
                graph[dep].append(component)
                in_degree[component] += 1
        
        # Initialize in-degrees
        for node in all_nodes:
            if node not in in_degree:
                in_degree[node] = 0
        
        # Topological sort
        queue = deque([node for node in all_nodes if in_degree[node] == 0])
        sorted_order = []
        
        while queue:
            node = queue.popleft()
            sorted_order.append(node)
            
            for neighbor in graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return sorted_order
    
    # Sort components by dependencies
    sorted_components = topological_sort(component_dependencies)
    print(f"Validation order: {sorted_components}")
    
    # Validate components in dependency order
    validation_results = {}
    validated_components = []
    
    for component in sorted_components:
        print(f"\nValidating {component}...")
        
        # Check if dependencies are satisfied
        deps = component_dependencies.get(component, [])
        missing_deps = [dep for dep in deps if dep not in validated_components]
        
        if missing_deps:
            print(f"  Skipping {component}: missing dependencies {missing_deps}")
            validation_results[component] = {
                'skipped': True,
                'reason': f'Missing dependencies: {missing_deps}'
            }
            continue
        
        # Validate component
        result = orchestrator.validate_workspace(
            developer_id=developer_id,
            components=[component],
            validation_types=["alignment", "builder"],
            levels=[1, 2, 3, 4]
        )
        
        validation_results[component] = {
            'result': result,
            'passed': result.overall_passed
        }
        
        if result.overall_passed:
            validated_components.append(component)
            print(f"  {component}: PASS")
        else:
            print(f"  {component}: FAIL")
            # Don't add to validated_components, affecting dependent components
    
    # Summary
    print(f"\nDependency-aware validation summary:")
    print(f"  Total components: {len(sorted_components)}")
    print(f"  Successfully validated: {len(validated_components)}")
    print(f"  Failed/Skipped: {len(sorted_components) - len(validated_components)}")
    
    return validation_results

# Usage
component_deps = {
    "data_loader": [],
    "feature_processor": ["data_loader"],
    "model_trainer": ["feature_processor"],
    "model_evaluator": ["model_trainer"],
    "pipeline_orchestrator": ["data_loader", "feature_processor", "model_trainer"]
}

dep_results = dependency_aware_validation_pattern(
    "/path/to/workspaces",
    "developer_1",
    component_deps
)
```

## Testing Integration Patterns

### Mock-Based Integration Testing

```python
from unittest.mock import Mock, patch
import tempfile
import os

def mock_based_integration_test():
    """Integration test using mocks for external dependencies"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Setup mock workspace structure
        workspace_root = temp_dir
        developer_id = "test_developer"
        
        # Create workspace structure
        workspace_path = os.path.join(workspace_root, developer_id)
        os.makedirs(workspace_path)
        
        for subdir in ["builders", "contracts", "scripts", "specs", "configs"]:
            os.makedirs(os.path.join(workspace_path, subdir))
        
        # Create mock files
        mock_builder = """
class TestBuilder:
    def __init__(self, config):
        self.config = config
    
    def build_step(self):
        return {"step_name": "test_step"}
"""
        
        with open(os.path.join(workspace_path, "builders", "test_builder.py"), "w") as f:
            f.write(mock_builder)
        
        # Mock configuration
        mock_config = {
            "workspace_name": "Test Workspace",
            "developer_id": developer_id,
            "version": "1.0.0",
            "workspace_settings": {
                "python_version": "3.9",
                "dependencies": [],
                "environment_variables": {},
                "custom_paths": [],
                "validation_rules": {}
            }
        }
        
        # Test integration with mocks
        with patch('cursus.validation.workspace.workspace_manager.WorkspaceManager.load_workspace_config') as mock_load:
            mock_load.return_value = Mock(**mock_config)
            
            # Initialize components
            manager = WorkspaceManager(workspace_root)
            orchestrator = WorkspaceValidationOrchestrator(workspace_root)
            
            # Test workspace discovery
            workspace_info = manager.get_workspace_info(developer_id)
            assert 'builders' in workspace_info
            
            # Test validation
            result = orchestrator.validate_workspace(
                developer_id=developer_id,
                components=["test_builder"],
                validation_types=["builder"]
            )
            
            print(f"Mock integration test: {'PASSED' if result.overall_passed else 'FAILED'}")
            return result.overall_passed

# Usage
test_passed = mock_based_integration_test()
```

### End-to-End Integration Test

```python
def end_to_end_integration_test(workspace_root):
    """Complete end-to-end integration test"""
    
    test_developer = "integration_test_dev"
    
    try:
        # Phase 1: Workspace Setup
        print("Phase 1: Workspace Setup")
        manager = WorkspaceManager(workspace_root)
        
        # Clean up any existing test workspace
        if manager.workspace_exists(test_developer):
            import shutil
            shutil.rmtree(os.path.join(workspace_root, test_developer))
        
        # Create workspace
        workspace_path = manager.create_workspace_structure(test_developer)
        print(f"  Created workspace: {workspace_path}")
        
        # Phase 2: Configuration Setup
        print("Phase 2: Configuration Setup")
        settings = WorkspaceSettings(
            python_version="3.9",
            dependencies=["numpy"],
            environment_variables={"TEST_MODE": "true"},
            validation_rules={"strict_typing": False}
        )
        
        config = WorkspaceConfig(
            workspace_name="Integration Test Workspace",
            developer_id=test_developer,
            version="1.0.0",
            description="End-to-end integration test workspace",
            workspace_settings=settings
        )
        
        manager.save_workspace_config(test_developer, config)
        print("  Configuration saved")
        
        # Phase 3: Component Creation
        print("Phase 3: Component Creation")
        
        # Create a simple builder
        builder_code = '''
class IntegrationTestBuilder:
    def __init__(self, config):
        self.config = config
    
    def build_step(self):
        return {
            "step_name": "integration_test_step",
            "step_type": "test",
            "config": self.config
        }
'''
        
        builder_path = os.path.join(workspace_path, "builders", "integration_test_builder.py")
        with open(builder_path, "w") as f:
            f.write(builder_code)
        print(f"  Created builder: {builder_path}")
        
        # Phase 4: Validation
        print("Phase 4: Validation")
        orchestrator = WorkspaceValidationOrchestrator(workspace_root)
        
        result = orchestrator.validate_workspace(
            developer_id=test_developer,
            components=["integration_test_builder"],
            validation_types=["builder"],
            levels=None
        )
        
        validation_passed = result.overall_passed
        print(f"  Validation: {'PASSED' if validation_passed else 'FAILED'}")
        
        # Phase 5: Cleanup
        print("Phase 5: Cleanup")
        import shutil
        shutil.rmtree(workspace_path)
        print("  Cleaned up test workspace")
        
        return validation_passed
        
    except Exception as e:
        print(f"Integration test failed: {e}")
        return False

# Usage
integration_success = end_to_end_integration_test("/path/to/workspaces")
print(f"End-to-end integration test: {'PASSED' if integration_success else 'FAILED'}")
```

## Performance Integration Patterns

### Parallel Validation Integration

```python
import concurrent.futures
import time

def parallel_validation_integration(workspace_root, developers, max_workers=4):
    """Integrate parallel validation across multiple developers"""
    
    def validate_single_developer(developer_id):
        """Validate a single developer's workspace"""
        start_time = time.time()
        
        orchestrator = WorkspaceValidationOrchestrator(workspace_root)
        
        result = orchestrator.validate_workspace(
            developer_id=developer_id,
            validation_types=["alignment", "builder"],
            levels=[1, 2, 3, 4]
        )
        
        end_time = time.time()
        
        return {
            'developer_id': developer_id,
            'result': result,
            'duration': end_time - start_time,
            'passed': result.overall_passed
        }
    
    # Sequential validation (baseline)
    print("Sequential validation:")
    sequential_start = time.time()
    sequential_results = []
    
    for developer_id in developers:
        result = validate_single_developer(developer_id)
        sequential_results.append(result)
        print(f"  {developer_id}: {'PASS' if result['passed'] else 'FAIL'} ({result['duration']:.2f}s)")
    
    sequential_total = time.time() - sequential_start
    print(f"Sequential total time: {sequential_total:.2f}s")
    
    # Parallel validation
    print(f"\nParallel validation (max_workers={max_workers}):")
    parallel_start = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_developer = {
            executor.submit(validate_single_developer, dev_id): dev_id 
            for dev_id in developers
        }
        
        parallel_results = []
        for future in concurrent.futures.as_completed(future_to_developer):
            result = future.result()
            parallel_results.append(result)
            print(f"  {result['developer_id']}: {'PASS' if result['passed'] else 'FAIL'} ({result['duration']:.2f}s)")
    
    parallel_total = time.time() - parallel_start
    print(f"Parallel total time: {parallel_total:.2f}s")
    
    # Performance comparison
    speedup = sequential_total / parallel_total if parallel_total > 0 else 0
    print(f"\nPerformance improvement: {speedup:.2f}x speedup")
    
    return {
        'sequential_results': sequential_results,
        'parallel_results': parallel_results,
        'sequential_time': sequential_total,
        'parallel_time': parallel_total,
        'speedup': speedup
    }

# Usage
developers = ["developer_1", "developer_2", "developer_3", "developer_4"]
perf_results = parallel_validation_integration("/path/to/workspaces", developers)
```

### Caching Integration Pattern

```python
from functools import lru_cache
import hashlib
import json

class CachedValidationIntegration:
    """Integration pattern with validation result caching"""
    
    def __init__(self, workspace_root, cache_size=128):
        self.workspace_root = workspace_root
        self.orchestrator = WorkspaceValidationOrchestrator(workspace_root)
        self.cache_size = cache_size
        
        # Setup caching
        self._validate_with_cache = lru_cache(maxsize=cache_size)(self._validate_workspace_impl)
    
    def _compute_cache_key(self, developer_id, components, validation_types, levels):
        """Compute cache key for validation parameters"""
        
        # Get workspace modification time
        manager = WorkspaceManager(self.workspace_root)
        workspace_info = manager.get_workspace_info(developer_id)
        
        # Create cache key from parameters and workspace state
        cache_data = {
            'developer_id': developer_id,
            'components': sorted(components) if components else None,
            'validation_types': sorted(validation_types),
            'levels': sorted(levels) if levels else None,
            'workspace_info': workspace_info
        }
        
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _validate_workspace_impl(self, cache_key, developer_id, components, validation_types, levels):
        """Internal validation implementation (cached)"""
        
        print(f"Cache miss for {developer_id} - running validation")
        
        validation_params = {
            'developer_id': developer_id,
            'validation_types': validation_types
        }
        
        if components:
            validation_params['components'] = components
        if levels:
            validation_params['levels'] = levels
        
        return self.orchestrator.validate_workspace(**validation_params)
    
    def validate_workspace_cached(self, developer_id, components=None, validation_types=None, levels=None):
        """Validate workspace with caching"""
        
        validation_types = validation_types or ["alignment", "builder"]
        levels = levels or [1, 2, 3, 4]
        
        # Compute cache key
        cache_key = self._compute_cache_key(developer_id, components, validation_types, levels)
        
        # Check cache
        try:
            result = self._validate_with_cache(cache_key, developer_id, components, validation_types, levels)
            print(f"Cache hit for {developer_id}")
            return result
        except Exception as e:
            print(f"Cache error for {developer_id}: {e}")
            # Fallback to direct validation
            return self.orchestrator.validate_workspace(
                developer_id=developer_id,
                components=components,
                validation_types=validation_types,
                levels=levels
            )
    
    def clear_cache(self):
        """Clear validation cache"""
        self._validate_with_cache.cache_clear()
        print("Validation cache cleared")
    
    def get_cache_info(self):
        """Get cache statistics"""
        return self._validate_with_cache.cache_info()

# Usage
cached_validator = CachedValidationIntegration("/path/to/workspaces")

# First validation (cache miss)
result1 = cached_validator.validate_workspace_cached("developer_1")

# Second validation (cache hit)
result2 = cached_validator.validate_workspace_cached("developer_1")

# Check cache statistics
cache_info = cached_validator.get_cache_info()
print(f"Cache stats: hits={cache_info.hits}, misses={cache_info.misses}")
```

## Best Practices for Integration

### 1. Graceful Degradation Pattern

```python
def graceful_degradation_validation(workspace_root, developer_id):
    """Validation with graceful degradation on failures"""
    
    orchestrator = WorkspaceValidationOrchestrator(workspace_root)
    
    # Define validation strategies in order of preference
    validation_strategies = [
        {
            'name': 'Full Validation',
            'params': {
                'validation_types': ['alignment', 'builder'],
                'levels': [1, 2, 3, 4]
            }
        },
        {
            'name': 'Alignment Only',
            'params': {
                'validation_types': ['alignment'],
                'levels': [1, 2, 3, 4]
            }
        },
        {
            'name': 'Basic Alignment',
            'params': {
                'validation_types': ['alignment'],
                'levels': [1, 2]
            }
        },
        {
            'name': 'Builder Only',
            'params': {
                'validation_types': ['builder']
            }
        }
    ]
    
    for strategy in validation_strategies:
        try:
            print(f"Attempting {strategy['name']}...")
            
            result = orchestrator.validate_workspace(
                developer_id=developer_id,
                **strategy['params']
            )
            
            if result.overall_passed:
                print(f"  {strategy['name']}: SUCCESS")
                return result
            else:
                print(f"  {strategy['name']}: FAILED - trying next strategy")
                
        except Exception as e:
            print(f"  {strategy['name']}: ERROR - {e}")
            continue
    
    print("All validation strategies failed")
    return None

# Usage
result = graceful_degradation_validation("/path/to/workspaces", "developer_1")
```

### 2. Health Check Integration

```python
def comprehensive_health_check(workspace_root, developer_id):
    """Comprehensive health check before validation"""
    
    health_checks = []
    
    # Check 1: Workspace existence
    manager = WorkspaceManager(workspace_root)
    if manager.workspace_exists(developer_id):
        health_checks.append(("Workspace exists", True, None))
    else:
        health_checks.append(("Workspace exists", False, "Workspace not found"))
        return health_checks  # Critical failure
    
    # Check 2: Configuration validity
    try:
        config = manager.load_workspace_config(developer_id)
        health_checks.append(("Configuration valid", True, None))
    except Exception as e:
        health_checks.append(("Configuration valid", False, str(e)))
        return health_checks  # Critical failure
    
    # Check 3: Component availability
    workspace_info = manager.get_workspace_info(developer_id)
    has_components = any(workspace_info.get(key, []) for key in ['builders', 'contracts', 'scripts'])
    health_checks.append(("Components available", has_components, 
                         None if has_components else "No components found"))
    
    # Check 4: Dependencies
    try:
        for dep in config.workspace_settings.dependencies:
            import importlib
            package_name = dep.split('>=')[0].split('==')[0]
            importlib.import_module(package_name)
        health_checks.append(("Dependencies satisfied", True, None))
    except ImportError as e:
        health_checks.append(("Dependencies satisfied", False, str(e)))
    
    # Check 5: File permissions
    import os
    workspace_path = os.path.join(workspace_root, developer_id)
    readable = os.access(workspace_path, os.R_OK)
    writable = os.access(workspace_path, os.W_OK)
    health_checks.append(("File permissions", readable and writable, 
                         None if readable and writable else "Insufficient file permissions"))
    
    return health_checks

# Usage
health_status = comprehensive_health_check("/path/to/workspaces", "developer_1")

print("Health Check Results:")
for check_name, passed, error in health_status:
    status = "PASS" if passed else "FAIL"
    print(f"  {check_name}: {status}")
    if error:
        print(f"    Error: {error}")
```

### 3. Circuit Breaker Pattern

```python
import time
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class ValidationCircuitBreaker:
    """Circuit breaker for validation operations"""
    
    def __init__(self, failure_threshold=5, recovery_timeout=60, success_threshold=3):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
    
    def call_validation(self, validation_func, *args, **kwargs):
        """Call validation function with circuit breaker protection"""
        
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                print("Circuit breaker: Attempting reset (HALF_OPEN)")
            else:
                raise Exception("Circuit breaker is OPEN - validation calls blocked")
        
        try:
            result = validation_func(*args, **kwargs)
            self._on_success(result)
            return result
        except Exception as e:
            self._on_failure(e)
            raise
    
    def _should_attempt_reset(self):
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self, result):
        """Handle successful validation"""
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                print("Circuit breaker: Reset to CLOSED")
        else:
            self.failure_count = 0
    
    def _on_failure(self, exception):
        """Handle validation failure"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            print(f"Circuit breaker: Opened due to {self.failure_count} failures")

def validation_with_circuit_breaker(workspace_root, developer_id):
    """Validation with circuit breaker protection"""
    
    circuit_breaker = ValidationCircuitBreaker(
        failure_threshold=3,
        recovery_timeout=30,
        success_threshold=2
    )
    
    orchestrator = WorkspaceValidationOrchestrator(workspace_root)
    
    def validate():
        return orchestrator.validate_workspace(
            developer_id=developer_id,
            validation_types=["alignment", "builder"],
            levels=[1, 2, 3, 4]
        )
    
    try:
        result = circuit_breaker.call_validation(validate)
        print(f"Validation completed: {'PASSED' if result.overall_passed else 'FAILED'}")
        return result
    except Exception as e:
        print(f"Validation blocked by circuit breaker: {e}")
        return None

# Usage
result = validation_with_circuit_breaker("/path/to/workspaces", "developer_1")
```

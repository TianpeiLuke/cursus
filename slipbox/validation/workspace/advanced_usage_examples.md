# Advanced Usage Examples

This document provides examples of advanced usage patterns for the workspace-aware validation system.

## Advanced Workspace Management

### Dynamic Workspace Creation and Migration

```python
from cursus.validation.workspace.workspace_manager import WorkspaceManager
from cursus.validation.workspace.models import WorkspaceConfig, WorkspaceSettings
import json
import shutil
from pathlib import Path

class AdvancedWorkspaceManager:
    """Advanced workspace management with migration and templating"""
    
    def __init__(self, workspace_root):
        self.workspace_root = Path(workspace_root)
        self.manager = WorkspaceManager(workspace_root)
    
    def create_workspace_from_blueprint(self, developer_id, blueprint_path, customizations=None):
        """Create workspace from a blueprint directory"""
        
        blueprint = Path(blueprint_path)
        if not blueprint.exists():
            raise ValueError(f"Blueprint not found: {blueprint_path}")
        
        # Create workspace structure
        workspace_path = self.manager.create_workspace_structure(developer_id)
        
        # Copy blueprint files
        for item in blueprint.rglob("*"):
            if item.is_file() and not item.name.startswith('.'):
                relative_path = item.relative_to(blueprint)
                target_path = Path(workspace_path) / relative_path
                
                # Create parent directories
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy and customize file
                content = item.read_text()
                if customizations:
                    for key, value in customizations.items():
                        content = content.replace(f"{{{{ {key} }}}}", str(value))
                
                target_path.write_text(content)
        
        print(f"Created workspace from blueprint: {workspace_path}")
        return workspace_path
    
    def migrate_workspace_structure(self, developer_id, target_version="2.0"):
        """Migrate workspace structure to new version"""
        
        workspace_path = Path(self.workspace_root) / developer_id
        if not workspace_path.exists():
            raise ValueError(f"Workspace not found: {developer_id}")
        
        # Backup current workspace
        backup_path = workspace_path.with_suffix(f".backup_{target_version}")
        shutil.copytree(workspace_path, backup_path)
        
        try:
            if target_version == "2.0":
                self._migrate_to_v2(workspace_path)
            
            print(f"Migrated {developer_id} to version {target_version}")
            return True
            
        except Exception as e:
            # Restore from backup on failure
            shutil.rmtree(workspace_path)
            shutil.move(backup_path, workspace_path)
            print(f"Migration failed, restored from backup: {e}")
            return False
    
    def _migrate_to_v2(self, workspace_path):
        """Migrate to version 2.0 structure"""
        
        # Add new directories
        new_dirs = ["templates", "tests", "docs"]
        for new_dir in new_dirs:
            (workspace_path / new_dir).mkdir(exist_ok=True)
        
        # Move old files if they exist
        old_config = workspace_path / "config.json"
        if old_config.exists():
            new_config = workspace_path / "workspace_config.json"
            shutil.move(old_config, new_config)
        
        # Update configuration format
        config_file = workspace_path / "workspace_config.json"
        if config_file.exists():
            config_data = json.loads(config_file.read_text())
            config_data["version"] = "2.0"
            config_data["migration_date"] = "2024-01-01T00:00:00Z"
            config_file.write_text(json.dumps(config_data, indent=2))

# Usage
advanced_manager = AdvancedWorkspaceManager("/path/to/workspaces")

# Create from blueprint
customizations = {
    "DEVELOPER_NAME": "John Doe",
    "PROJECT_NAME": "ML Pipeline",
    "PYTHON_VERSION": "3.9"
}

workspace_path = advanced_manager.create_workspace_from_blueprint(
    "developer_1", 
    "/path/to/blueprints/ml_template",
    customizations
)

# Migrate workspace
migration_success = advanced_manager.migrate_workspace_structure("developer_1", "2.0")
```

### Multi-Environment Workspace Orchestration

```python
from cursus.validation.workspace.workspace_orchestrator import WorkspaceValidationOrchestrator
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class EnvironmentConfig:
    name: str
    workspace_root: str
    validation_types: List[str]
    levels: List[int]
    parallel: bool = True
    max_workers: int = 4

class MultiEnvironmentOrchestrator:
    """Orchestrate validation across multiple environments"""
    
    def __init__(self, environments: Dict[str, EnvironmentConfig]):
        self.environments = environments
        self.orchestrators = {
            name: WorkspaceValidationOrchestrator(config.workspace_root)
            for name, config in environments.items()
        }
    
    async def validate_across_environments(self, developer_id: str, components: Optional[List[str]] = None):
        """Validate developer across all environments"""
        
        async def validate_environment(env_name: str, config: EnvironmentConfig):
            """Validate in a single environment"""
            
            orchestrator = self.orchestrators[env_name]
            
            # Run validation in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                result = await loop.run_in_executor(
                    executor,
                    orchestrator.validate_workspace,
                    developer_id,
                    components,
                    config.validation_types,
                    config.levels,
                    config.parallel,
                    config.max_workers
                )
            
            return {
                'environment': env_name,
                'result': result,
                'passed': result.overall_passed
            }
        
        # Run validations concurrently across environments
        tasks = [
            validate_environment(env_name, config)
            for env_name, config in self.environments.items()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        environment_results = {}
        for result in results:
            if isinstance(result, Exception):
                print(f"Environment validation failed: {result}")
                continue
            
            env_name = result['environment']
            environment_results[env_name] = result
            
            status = "PASS" if result['passed'] else "FAIL"
            print(f"{env_name}: {status}")
        
        return environment_results
    
    def promote_across_environments(self, developer_id: str, source_env: str, target_env: str):
        """Promote workspace from source to target environment"""
        
        if source_env not in self.environments or target_env not in self.environments:
            raise ValueError("Invalid environment specified")
        
        source_config = self.environments[source_env]
        target_config = self.environments[target_env]
        
        # Validate source environment first
        source_orchestrator = self.orchestrators[source_env]
        source_result = source_orchestrator.validate_workspace(
            developer_id=developer_id,
            validation_types=source_config.validation_types,
            levels=source_config.levels
        )
        
        if not source_result.overall_passed:
            print(f"Source environment validation failed: {source_env}")
            return False
        
        # Copy workspace configuration and components
        source_manager = WorkspaceManager(source_config.workspace_root)
        target_manager = WorkspaceManager(target_config.workspace_root)
        
        try:
            # Load source configuration
            source_workspace_config = source_manager.load_workspace_config(developer_id)
            
            # Create target workspace if it doesn't exist
            if not target_manager.workspace_exists(developer_id):
                target_manager.create_workspace_structure(developer_id)
            
            # Adapt configuration for target environment
            target_workspace_config = source_workspace_config.copy(deep=True)
            target_workspace_config.description = f"Promoted from {source_env} to {target_env}"
            
            # Save to target environment
            target_manager.save_workspace_config(developer_id, target_workspace_config)
            
            # Validate target environment
            target_orchestrator = self.orchestrators[target_env]
            target_result = target_orchestrator.validate_workspace(
                developer_id=developer_id,
                validation_types=target_config.validation_types,
                levels=target_config.levels
            )
            
            if target_result.overall_passed:
                print(f"Successfully promoted {developer_id} from {source_env} to {target_env}")
                return True
            else:
                print(f"Target environment validation failed: {target_env}")
                return False
                
        except Exception as e:
            print(f"Promotion failed: {e}")
            return False

# Usage
environments = {
    "development": EnvironmentConfig(
        name="development",
        workspace_root="/workspaces/dev",
        validation_types=["builder"],
        levels=[1, 2],
        parallel=True,
        max_workers=2
    ),
    "staging": EnvironmentConfig(
        name="staging",
        workspace_root="/workspaces/staging",
        validation_types=["alignment", "builder"],
        levels=[1, 2, 3],
        parallel=True,
        max_workers=4
    ),
    "production": EnvironmentConfig(
        name="production",
        workspace_root="/workspaces/prod",
        validation_types=["alignment", "builder"],
        levels=[1, 2, 3, 4],
        parallel=True,
        max_workers=8
    )
}

multi_orchestrator = MultiEnvironmentOrchestrator(environments)

# Async validation across environments
async def run_multi_env_validation():
    results = await multi_orchestrator.validate_across_environments("developer_1")
    return results

# Run async validation
# results = asyncio.run(run_multi_env_validation())

# Promote between environments
promotion_success = multi_orchestrator.promote_across_environments(
    "developer_1", "development", "staging"
)
```

## Advanced Validation Patterns

### Custom Validation Rules Engine

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ValidationRule:
    name: str
    description: str
    severity: str  # "error", "warning", "info"
    enabled: bool = True

class ValidationRuleEngine(ABC):
    """Abstract base for validation rule engines"""
    
    @abstractmethod
    def evaluate(self, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Evaluate rule and return (passed, messages)"""
        pass

class CodeQualityRuleEngine(ValidationRuleEngine):
    """Rule engine for code quality validation"""
    
    def __init__(self, rules: List[ValidationRule]):
        self.rules = {rule.name: rule for rule in rules if rule.enabled}
    
    def evaluate(self, context: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Evaluate code quality rules"""
        
        messages = []
        all_passed = True
        
        # Rule: Check for docstrings
        if "require_docstrings" in self.rules:
            if not self._check_docstrings(context):
                messages.append("Missing docstrings in some functions")
                if self.rules["require_docstrings"].severity == "error":
                    all_passed = False
        
        # Rule: Check complexity
        if "max_complexity" in self.rules:
            complexity_issues = self._check_complexity(context)
            if complexity_issues:
                messages.extend(complexity_issues)
                if self.rules["max_complexity"].severity == "error":
                    all_passed = False
        
        # Rule: Check type hints
        if "require_type_hints" in self.rules:
            if not self._check_type_hints(context):
                messages.append("Missing type hints in some functions")
                if self.rules["require_type_hints"].severity == "error":
                    all_passed = False
        
        return all_passed, messages
    
    def _check_docstrings(self, context: Dict[str, Any]) -> bool:
        """Check if functions have docstrings"""
        # Simplified implementation
        module_content = context.get('module_content', '')
        return '"""' in module_content or "'''" in module_content
    
    def _check_complexity(self, context: Dict[str, Any]) -> List[str]:
        """Check cyclomatic complexity"""
        # Simplified implementation
        module_content = context.get('module_content', '')
        issues = []
        
        # Count nested structures as complexity indicator
        nesting_level = 0
        max_nesting = 0
        
        for line in module_content.split('\n'):
            stripped = line.strip()
            if stripped.startswith(('if ', 'for ', 'while ', 'try:', 'with ')):
                nesting_level += 1
                max_nesting = max(max_nesting, nesting_level)
            elif stripped in ('else:', 'elif ', 'except:', 'finally:'):
                continue
            elif stripped == '' or stripped.startswith('#'):
                continue
            else:
                # Reset nesting on function/class definition
                if stripped.startswith(('def ', 'class ')):
                    nesting_level = 0
        
        if max_nesting > 4:  # Arbitrary threshold
            issues.append(f"High complexity detected (nesting level: {max_nesting})")
        
        return issues
    
    def _check_type_hints(self, context: Dict[str, Any]) -> bool:
        """Check if functions have type hints"""
        # Simplified implementation
        module_content = context.get('module_content', '')
        return '->' in module_content and ':' in module_content

class AdvancedWorkspaceValidator:
    """Advanced validator with custom rule engines"""
    
    def __init__(self, workspace_root: str):
        self.workspace_root = workspace_root
        self.orchestrator = WorkspaceValidationOrchestrator(workspace_root)
        self.rule_engines = {}
    
    def add_rule_engine(self, name: str, engine: ValidationRuleEngine):
        """Add a custom rule engine"""
        self.rule_engines[name] = engine
    
    def validate_with_custom_rules(self, developer_id: str, components: List[str] = None):
        """Validate workspace with custom rules"""
        
        # Standard validation first
        standard_result = self.orchestrator.validate_workspace(
            developer_id=developer_id,
            components=components,
            validation_types=["alignment", "builder"],
            levels=[1, 2, 3, 4]
        )
        
        # Custom rule validation
        custom_results = {}
        
        for component in (components or self._discover_components(developer_id)):
            component_results = {}
            
            # Load component context
            context = self._load_component_context(developer_id, component)
            
            # Apply custom rule engines
            for engine_name, engine in self.rule_engines.items():
                passed, messages = engine.evaluate(context)
                component_results[engine_name] = {
                    'passed': passed,
                    'messages': messages
                }
            
            custom_results[component] = component_results
        
        # Combine results
        overall_passed = standard_result.overall_passed and all(
            all(result['passed'] for result in component_results.values())
            for component_results in custom_results.values()
        )
        
        return {
            'standard_validation': standard_result,
            'custom_validation': custom_results,
            'overall_passed': overall_passed
        }
    
    def _discover_components(self, developer_id: str) -> List[str]:
        """Discover components in workspace"""
        manager = WorkspaceManager(self.workspace_root)
        workspace_info = manager.get_workspace_info(developer_id)
        return workspace_info.get('builders', [])
    
    def _load_component_context(self, developer_id: str, component: str) -> Dict[str, Any]:
        """Load context for component validation"""
        
        # Load component file content
        from cursus.validation.workspace.file_resolver import DeveloperWorkspaceFileResolver
        
        resolver = DeveloperWorkspaceFileResolver(self.workspace_root, developer_id)
        
        try:
            builder_path = resolver.resolve_builder_file(component)
            with open(builder_path, 'r') as f:
                module_content = f.read()
        except Exception:
            module_content = ""
        
        return {
            'component_name': component,
            'developer_id': developer_id,
            'module_content': module_content,
            'file_path': builder_path if 'builder_path' in locals() else None
        }

# Usage
# Define custom rules
quality_rules = [
    ValidationRule("require_docstrings", "All functions must have docstrings", "warning"),
    ValidationRule("max_complexity", "Functions should not be too complex", "error"),
    ValidationRule("require_type_hints", "Functions should have type hints", "info")
]

# Create rule engine
quality_engine = CodeQualityRuleEngine(quality_rules)

# Create advanced validator
advanced_validator = AdvancedWorkspaceValidator("/path/to/workspaces")
advanced_validator.add_rule_engine("code_quality", quality_engine)

# Run validation with custom rules
results = advanced_validator.validate_with_custom_rules("developer_1", ["xgboost_trainer"])

print(f"Overall validation: {'PASSED' if results['overall_passed'] else 'FAILED'}")
print(f"Standard validation: {'PASSED' if results['standard_validation'].overall_passed else 'FAILED'}")

for component, custom_results in results['custom_validation'].items():
    print(f"\nComponent: {component}")
    for engine_name, engine_result in custom_results.items():
        status = "PASS" if engine_result['passed'] else "FAIL"
        print(f"  {engine_name}: {status}")
        for message in engine_result['messages']:
            print(f"    - {message}")
```

## Advanced Error Handling and Recovery

### Resilient Validation with Auto-Recovery

```python
import time
import logging
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum

class RecoveryStrategy(Enum):
    RETRY = "retry"
    FALLBACK = "fallback"
    SKIP = "skip"
    ABORT = "abort"

@dataclass
class RecoveryConfig:
    max_retries: int = 3
    retry_delay: float = 1.0
    backoff_multiplier: float = 2.0
    strategy: RecoveryStrategy = RecoveryStrategy.RETRY
    fallback_function: Optional[Callable] = None
    recovery_actions: List[Callable] = field(default_factory=list)

class ResilientValidator:
    """Validator with advanced error handling and recovery"""
    
    def __init__(self, workspace_root: str, recovery_config: RecoveryConfig = None):
        self.workspace_root = workspace_root
        self.orchestrator = WorkspaceValidationOrchestrator(workspace_root)
        self.recovery_config = recovery_config or RecoveryConfig()
        self.logger = logging.getLogger(__name__)
    
    def validate_with_recovery(self, developer_id: str, components: List[str] = None):
        """Validate with automatic error recovery"""
        
        attempt = 0
        last_exception = None
        
        while attempt < self.recovery_config.max_retries:
            try:
                # Attempt validation
                result = self.orchestrator.validate_workspace(
                    developer_id=developer_id,
                    components=components,
                    validation_types=["alignment", "builder"],
                    levels=[1, 2, 3, 4]
                )
                
                if result.overall_passed:
                    self.logger.info(f"Validation succeeded on attempt {attempt + 1}")
                    return result
                else:
                    # Validation failed, try recovery actions
                    self.logger.warning(f"Validation failed on attempt {attempt + 1}")
                    self._execute_recovery_actions(developer_id, result)
                    
            except Exception as e:
                last_exception = e
                self.logger.error(f"Validation error on attempt {attempt + 1}: {e}")
                
                # Apply recovery strategy
                if self.recovery_config.strategy == RecoveryStrategy.RETRY:
                    if attempt < self.recovery_config.max_retries - 1:
                        delay = self.recovery_config.retry_delay * (
                            self.recovery_config.backoff_multiplier ** attempt
                        )
                        self.logger.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                elif self.recovery_config.strategy == RecoveryStrategy.FALLBACK:
                    if self.recovery_config.fallback_function:
                        return self.recovery_config.fallback_function(developer_id, components)
                elif self.recovery_config.strategy == RecoveryStrategy.SKIP:
                    self.logger.warning("Skipping validation due to error")
                    return None
                elif self.recovery_config.strategy == RecoveryStrategy.ABORT:
                    raise e
            
            attempt += 1
        
        # All retries exhausted
        self.logger.error(f"Validation failed after {self.recovery_config.max_retries} attempts")
        if last_exception:
            raise last_exception
        
        return None
    
    def _execute_recovery_actions(self, developer_id: str, failed_result):
        """Execute recovery actions after validation failure"""
        
        for action in self.recovery_config.recovery_actions:
            try:
                self.logger.info(f"Executing recovery action: {action.__name__}")
                action(developer_id, failed_result)
            except Exception as e:
                self.logger.error(f"Recovery action failed: {e}")

def cleanup_workspace_action(developer_id: str, failed_result):
    """Recovery action to clean up workspace"""
    print(f"Cleaning up workspace for {developer_id}")
    # Implementation would clean temporary files, reset state, etc.

def reset_configuration_action(developer_id: str, failed_result):
    """Recovery action to reset configuration"""
    print(f"Resetting configuration for {developer_id}")
    # Implementation would reset to default configuration

def fallback_validation(developer_id: str, components: List[str]):
    """Fallback validation with minimal requirements"""
    print(f"Running fallback validation for {developer_id}")
    # Simplified validation logic
    return type('Result', (), {'overall_passed': True, 'errors': []})()

# Usage
recovery_config = RecoveryConfig(
    max_retries=5,
    retry_delay=2.0,
    backoff_multiplier=1.5,
    strategy=RecoveryStrategy.RETRY,
    fallback_function=fallback_validation,
    recovery_actions=[cleanup_workspace_action, reset_configuration_action]
)

resilient_validator = ResilientValidator("/path/to/workspaces", recovery_config)

# Run resilient validation
try:
    result = resilient_validator.validate_with_recovery("developer_1")
    if result:
        print(f"Resilient validation: {'PASSED' if result.overall_passed else 'FAILED'}")
    else:
        print("Validation could not be completed")
except Exception as e:
    print(f"Validation failed permanently: {e}")
```

## Advanced Performance Optimization

### Intelligent Caching and Memoization

```python
import hashlib
import pickle
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from functools import wraps
import threading

class IntelligentCache:
    """Intelligent caching system with dependency tracking"""
    
    def __init__(self, cache_dir: str, max_size: int = 1000, ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.ttl = ttl
        self.access_times = {}
        self.dependencies = {}
        self.lock = threading.RLock()
    
    def _compute_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Compute cache key from function signature"""
        key_data = {
            'function': func_name,
            'args': args,
            'kwargs': kwargs
        }
        key_str = pickle.dumps(key_data, protocol=pickle.HIGHEST_PROTOCOL)
        return hashlib.sha256(key_str).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key"""
        return self.cache_dir / f"{key}.cache"
    
    def _is_valid(self, cache_path: Path, dependencies: List[str] = None) -> bool:
        """Check if cache entry is still valid"""
        if not cache_path.exists():
            return False
        
        # Check TTL
        cache_time = cache_path.stat().st_mtime
        if time.time() - cache_time > self.ttl:
            return False
        
        # Check dependencies
        if dependencies:
            for dep_path in dependencies:
                dep_file = Path(dep_path)
                if dep_file.exists() and dep_file.stat().st_mtime > cache_time:
                    return False
        
        return True
    
    def get(self, key: str, dependencies: List[str] = None) -> Optional[Any]:
        """Get cached value"""
        with self.lock:
            cache_path = self._get_cache_path(key)
            
            if self._is_valid(cache_path, dependencies):
                try:
                    with open(cache_path, 'rb') as f:
                        result = pickle.load(f)
                    
                    # Update access time
                    self.access_times[key] = time.time()
                    return result
                except Exception:
                    # Cache corrupted, remove it
                    cache_path.unlink(missing_ok=True)
            
            return None
    
    def set(self, key: str, value: Any, dependencies: List[str] = None):
        """Set cached value"""
        with self.lock:
            # Enforce cache size limit
            if len(self.access_times) >= self.max_size:
                self._evict_oldest()
            
            cache_path = self._get_cache_path(key)
            
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                self.access_times[key] = time.time()
                if dependencies:
                    self.dependencies[key] = dependencies
                    
            except Exception as e:
                print(f"Failed to cache result: {e}")
    
    def _evict_oldest(self):
        """Evict oldest cache entries"""
        if not self.access_times:
            return
        
        # Remove 10% of oldest entries
        num_to_remove = max(1, len(self.access_times) // 10)
        oldest_keys = sorted(self.access_times.keys(), 
                           key=lambda k: self.access_times[k])[:num_to_remove]
        
        for key in oldest_keys:
            cache_path = self._get_cache_path(key)
            cache_path.unlink(missing_ok=True)
            del self.access_times[key]
            self.dependencies.pop(key, None)
    
    def invalidate(self, pattern: str = None):
        """Invalidate cache entries"""
        with self.lock:
            if pattern:
                # Invalidate matching patterns
                keys_to_remove = [k for k in self.access_times.keys() if pattern in k]
            else:
                # Invalidate all
                keys_to_remove = list(self.access_times.keys())
            
            for key in keys_to_remove:
                cache_path = self._get_cache_path(key)
                cache_path.unlink(missing_ok=True)
                del self.access_times[key]
                self.dependencies.pop(key, None)

def intelligent_cache(cache_instance: IntelligentCache, dependencies_func: Callable = None):
    """Decorator for intelligent caching"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Compute cache key
            key = cache_instance._compute_key(func.__name__, args, kwargs)
            
            # Get dependencies if function provided
            dependencies = []
            if dependencies_func:
                try:
                    dependencies = dependencies_func(*args, **kwargs)
                except Exception:
                    pass
            
            # Try to get from cache
            cached_result = cache_instance.get(key, dependencies)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            cache_instance.set(key, result, dependencies)
            
            return result
        
        return wrapper
    return decorator

class OptimizedWorkspaceValidator:
    """Workspace validator with intelligent caching"""
    
    def __init__(self, workspace_root: str, cache_dir: str = None):
        self.workspace_root = workspace_root
        self.orchestrator = WorkspaceValidationOrchestrator(workspace_root)
        
        # Setup intelligent cache
        cache_dir = cache_dir or f"{workspace_root}/.cache"
        self.cache = IntelligentCache(cache_dir, max_size=500, ttl=1800)  # 30 min TTL
    
    def _get_validation_dependencies(self, developer_id: str, components: List[str] = None):
        """Get file dependencies for validation caching"""
        dependencies = []
        
        try:
            from cursus.validation.workspace.file_resolver import DeveloperWorkspaceFileResolver
            resolver = DeveloperWorkspaceFileResolver(self.workspace_root, developer_id)
            
            if components:
                for component in components:
                    try:
                        # Add component files as dependencies
                        builder_path = resolver.resolve_builder_file(component)
                        dependencies.append(builder_path)
                        
                        contract_path = resolver.resolve_contract_file(component)
                        dependencies.append(contract_path)
                        
                        script_path = resolver.resolve_script_file(component)
                        dependencies.append(script_path)
                        
                    except Exception:
                        continue
            
            # Add workspace config as dependency
            workspace_path = Path(self.workspace_root) / developer_id
            config_path = workspace_path / "workspace_config.json"
            if config_path.exists():
                dependencies.append(str(config_path))
                
        except Exception:
            pass
        
        return dependencies
    
    @intelligent_cache(cache_instance=self.cache, dependencies_func=self._get_validation_dependencies)
    def validate_workspace_cached(self, developer_id: str, components: List[str] = None, 
                                validation_types: List[str] = None, levels: List[int] = None):
        """Cached workspace validation with dependency tracking"""
        
        validation_types = validation_types or ["alignment", "builder"]
        levels = levels or [1, 2, 3, 4]
        
        return self.orchestrator.validate_workspace(
            developer_id=developer_id,
            components=components,
            validation_types=validation_types,
            levels=levels
        )
    
    def clear_cache(self, pattern: str = None):
        """Clear validation cache"""
        self.cache.invalidate(pattern)
        print(f"Cache cleared{f' (pattern: {pattern})' if pattern else ''}")
    
    def get_cache_stats(self):
        """Get cache statistics"""
        return {
            'cache_size': len(self.cache.access_times),
            'max_size': self.cache.max_size,
            'ttl': self.cache.ttl
        }

# Usage
optimized_validator = OptimizedWorkspaceValidator("/path/to/workspaces")

# First validation (cache miss)
print("First validation (cache miss):")
result1 = optimized_validator.validate_workspace_cached("developer_1", ["xgboost_trainer"])

# Second validation (cache hit)
print("\nSecond validation (cache hit):")
result2 = optimized_validator.validate_workspace_cached("developer_1", ["xgboost_trainer"])

# Check cache stats
stats = optimized_validator.get_cache_stats()
print(f"\nCache stats: {stats}")

# Clear cache for specific pattern
optimized_validator.clear_cache("developer_1")
```

## Advanced Monitoring and Analytics

### Validation Metrics Collection

```python
import time
import json
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from typing import Dict, List, Any
from dataclasses import dataclass, asdict

@dataclass
class ValidationMetrics:
    timestamp: str
    developer_id: str
    components: List[str]
    validation_types: List[str]
    levels: List[int]
    duration: float
    passed: bool
    error_count: int
    warning_count: int
    details: Dict[str, Any]

class ValidationAnalytics:
    """Advanced analytics for validation operations"""
    
    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or "validation_metrics.json"
        self.metrics_history = []
        self.load_metrics()
    
    def record_validation(self, developer_id: str, components: List[str], 
                         validation_types: List[str], levels: List[int],
                         duration: float, result, details: Dict[str, Any] = None):
        """Record validation metrics"""
        
        metrics = ValidationMetrics(
            timestamp=datetime.now().isoformat(),
            developer_id=developer_id,
            components=components or [],
            validation_types=validation_types,
            levels=levels,
            duration=duration,
            passed=result.overall_passed if result else False,
            error_count=len(result.errors) if result and hasattr(result, 'errors') else 0,
            warning_count=0,  # Would need to extract from result
            details=details or {}
        )
        
        self.metrics_history.append(metrics)
        self.save_metrics()
    
    def get_success_rate(self, developer_id: str = None, days: int = 7) -> float:
        """Get validation success rate"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        relevant_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m.timestamp) >= cutoff_date
            and (developer_id is None or m.developer_id == developer_id)
        ]
        
        if not relevant_metrics:
            return 0.0
        
        passed_count = sum(1 for m in relevant_metrics if m.passed)
        return passed_count / len(relevant_metrics)
    
    def get_performance_trends(self, developer_id: str = None, days: int = 30):
        """Get performance trends over time"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        relevant_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m.timestamp) >= cutoff_date
            and (developer_id is None or m.developer_id == developer_id)
        ]
        
        # Group by day
        daily_metrics = defaultdict(list)
        for metric in relevant_metrics:
            day = datetime.fromisoformat(metric.timestamp).date()
            daily_metrics[day].append(metric)
        
        trends = {}
        for day, day_metrics in daily_metrics.items():
            trends[day.isoformat()] = {
                'total_validations': len(day_metrics),
                'success_rate': sum(1 for m in day_metrics if m.passed) / len(day_metrics),
                'avg_duration': sum(m.duration for m in day_metrics) / len(day_metrics),
                'error_rate': sum(m.error_count for m in day_metrics) / len(day_metrics)
            }
        
        return trends
    
    def get_component_analysis(self, days: int = 30):
        """Analyze validation results by component"""
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        relevant_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m.timestamp) >= cutoff_date
        ]
        
        component_stats = defaultdict(lambda: {
            'total_validations': 0,
            'passed_validations': 0,
            'total_duration': 0.0,
            'error_count': 0
        })
        
        for metric in relevant_metrics:
            for component in metric.components:
                stats = component_stats[component]
                stats['total_validations'] += 1
                if metric.passed:
                    stats['passed_validations'] += 1
                stats['total_duration'] += metric.duration
                stats['error_count'] += metric.error_count
        
        # Calculate derived metrics
        analysis = {}
        for component, stats in component_stats.items():
            analysis[component] = {
                'success_rate': stats['passed_validations'] / stats['total_validations'],
                'avg_duration': stats['total_duration'] / stats['total_validations'],
                'avg_errors': stats['error_count'] / stats['total_validations'],
                'total_validations': stats['total_validations']
            }
        
        return analysis
    
    def generate_report(self, developer_id: str = None, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        success_rate = self.get_success_rate(developer_id, days)
        trends = self.get_performance_trends(developer_id, days)
        component_analysis = self.get_component_analysis(days)
        
        # Get recent metrics for summary
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m.timestamp) >= cutoff_date
            and (developer_id is None or m.developer_id == developer_id)
        ]
        
        report = {
            'report_period': f"Last {days} days",
            'developer_id': developer_id or "All developers",
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_validations': len(recent_metrics),
                'success_rate': success_rate,
                'avg_duration': sum(m.duration for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0,
                'total_errors': sum(m.error_count for m in recent_metrics)
            },
            'trends': trends,
            'component_analysis': component_analysis,
            'recommendations': self._generate_recommendations(recent_metrics, component_analysis)
        }
        
        return report
    
    def _generate_recommendations(self, metrics: List[ValidationMetrics], 
                                component_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on metrics"""
        
        recommendations = []
        
        if not metrics:
            return ["No recent validation data available"]
        
        # Success rate recommendations
        success_rate = sum(1 for m in metrics if m.passed) / len(metrics)
        if success_rate < 0.8:
            recommendations.append("Success rate is below 80%. Consider reviewing failing components.")
        
        # Performance recommendations
        avg_duration = sum(m.duration for m in metrics) / len(metrics)
        if avg_duration > 60:  # More than 1 minute
            recommendations.append("Average validation time is high. Consider optimizing validation logic.")
        
        # Component-specific recommendations
        for component, analysis in component_analysis.items():
            if analysis['success_rate'] < 0.7:
                recommendations.append(f"Component '{component}' has low success rate ({analysis['success_rate']:.1%})")
            
            if analysis['avg_duration'] > 30:
                recommendations.append(f"Component '{component}' has high validation time ({analysis['avg_duration']:.1f}s)")
        
        return recommendations
    
    def save_metrics(self):
        """Save metrics to storage"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump([asdict(m) for m in self.metrics_history], f, indent=2)
        except Exception as e:
            print(f"Failed to save metrics: {e}")
    
    def load_metrics(self):
        """Load metrics from storage"""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
                self.metrics_history = [ValidationMetrics(**item) for item in data]
        except FileNotFoundError:
            self.metrics_history = []
        except Exception as e:
            print(f"Failed to load metrics: {e}")
            self.metrics_history = []

class AnalyticsEnabledValidator:
    """Validator with built-in analytics"""
    
    def __init__(self, workspace_root: str, analytics_storage: str = None):
        self.workspace_root = workspace_root
        self.orchestrator = WorkspaceValidationOrchestrator(workspace_root)
        self.analytics = ValidationAnalytics(analytics_storage)
    
    def validate_with_analytics(self, developer_id: str, components: List[str] = None,
                              validation_types: List[str] = None, levels: List[int] = None):
        """Validate workspace and record analytics"""
        
        validation_types = validation_types or ["alignment", "builder"]
        levels = levels or [1, 2, 3, 4]
        
        start_time = time.time()
        
        try:
            result = self.orchestrator.validate_workspace(
                developer_id=developer_id,
                components=components,
                validation_types=validation_types,
                levels=levels
            )
            
            duration = time.time() - start_time
            
            # Record metrics
            self.analytics.record_validation(
                developer_id=developer_id,
                components=components or [],
                validation_types=validation_types,
                levels=levels,
                duration=duration,
                result=result,
                details={'workspace_root': self.workspace_root}
            )
            
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Record failed validation
            self.analytics.record_validation(
                developer_id=developer_id,
                components=components or [],
                validation_types=validation_types,
                levels=levels,
                duration=duration,
                result=None,
                details={'error': str(e), 'workspace_root': self.workspace_root}
            )
            
            raise
    
    def get_developer_report(self, developer_id: str, days: int = 7):
        """Get analytics report for specific developer"""
        return self.analytics.generate_report(developer_id, days)
    
    def get_overall_report(self, days: int = 7):
        """Get overall analytics report"""
        return self.analytics.generate_report(None, days)

# Usage
analytics_validator = AnalyticsEnabledValidator("/path/to/workspaces")

# Run validations with analytics
for i in range(5):
    result = analytics_validator.validate_with_analytics(
        f"developer_{i % 3 + 1}", 
        ["xgboost_trainer", "data_processor"]
    )
    print(f"Validation {i+1}: {'PASSED' if result.overall_passed else 'FAILED'}")

# Generate reports
developer_report = analytics_validator.get_developer_report("developer_1")
overall_report = analytics_validator.get_overall_report()

print(f"\nDeveloper 1 Success Rate: {developer_report['summary']['success_rate']:.1%}")
print(f"Overall Success Rate: {overall_report['summary']['success_rate']:.1%}")

print("\nRecommendations:")
for rec in overall_report['recommendations']:
    print(f"  - {rec}")
```

## Best Practices Summary

### 1. Production-Ready Validation Pipeline

```python
def create_production_validation_pipeline(workspace_root: str):
    """Create a production-ready validation pipeline"""
    
    # Setup components
    cache_dir = f"{workspace_root}/.validation_cache"
    analytics_storage = f"{workspace_root}/.validation_analytics.json"
    
    # Create optimized validator with caching
    validator = OptimizedWorkspaceValidator(workspace_root, cache_dir)
    
    # Create analytics-enabled validator
    analytics_validator = AnalyticsEnabledValidator(workspace_root, analytics_storage)
    
    # Setup resilient validation
    recovery_config = RecoveryConfig(
        max_retries=3,
        retry_delay=1.0,
        backoff_multiplier=2.0,
        strategy=RecoveryStrategy.RETRY
    )
    resilient_validator = ResilientValidator(workspace_root, recovery_config)
    
    # Create multi-environment orchestrator
    environments = {
        "development": EnvironmentConfig(
            name="development",
            workspace_root=f"{workspace_root}/dev",
            validation_types=["builder"],
            levels=[1, 2]
        ),
        "production": EnvironmentConfig(
            name="production", 
            workspace_root=f"{workspace_root}/prod",
            validation_types=["alignment", "builder"],
            levels=[1, 2, 3, 4]
        )
    }
    multi_orchestrator = MultiEnvironmentOrchestrator(environments)
    
    return {
        'optimized_validator': validator,
        'analytics_validator': analytics_validator,
        'resilient_validator': resilient_validator,
        'multi_orchestrator': multi_orchestrator
    }

# Usage
pipeline = create_production_validation_pipeline("/path/to/workspaces")

# Use different validators based on needs
cached_result = pipeline['optimized_validator'].validate_workspace_cached("developer_1")
analytics_result = pipeline['analytics_validator'].validate_with_analytics("developer_1")
resilient_result = pipeline['resilient_validator'].validate_with_recovery("developer_1")
```

### 2. Monitoring and Alerting

```python
def setup_validation_monitoring(analytics_validator: AnalyticsEnabledValidator):
    """Setup monitoring and alerting for validation system"""
    
    def check_system_health():
        """Check overall system health"""
        report = analytics_validator.get_overall_report(days=1)  # Last 24 hours
        
        alerts = []
        
        # Check success rate
        if report['summary']['success_rate'] < 0.9:
            alerts.append(f"Low success rate: {report['summary']['success_rate']:.1%}")
        
        # Check average duration
        if report['summary']['avg_duration'] > 120:  # 2 minutes
            alerts.append(f"High validation time: {report['summary']['avg_duration']:.1f}s")
        
        # Check error rate
        error_rate = report['summary']['total_errors'] / max(report['summary']['total_validations'], 1)
        if error_rate > 0.1:  # More than 10% error rate
            alerts.append(f"High error rate: {error_rate:.1%}")
        
        return alerts
    
    def send_alerts(alerts: List[str]):
        """Send alerts (placeholder implementation)"""
        if alerts:
            print("VALIDATION SYSTEM ALERTS:")
            for alert in alerts:
                print(f"  🚨 {alert}")
        else:
            print("✅ Validation system healthy")
    
    # Check health and send alerts
    alerts = check_system_health()
    send_alerts(alerts)
    
    return alerts

# Usage
alerts = setup_validation_monitoring(analytics_validator)
```

This completes the advanced usage examples with comprehensive patterns for production deployment, monitoring, and analytics.

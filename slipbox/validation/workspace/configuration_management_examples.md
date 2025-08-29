# Configuration Management Examples

This document provides examples of using the workspace-aware configuration management system.

## Basic Configuration Management

### Using WorkspaceManager for Configuration

```python
from cursus.validation.workspace.workspace_manager import WorkspaceManager

# Initialize workspace manager
manager = WorkspaceManager(workspace_root="/path/to/workspaces")

# Load workspace configuration
config = manager.load_workspace_config("developer_1")

print(f"Workspace name: {config.workspace_name}")
print(f"Developer ID: {config.developer_id}")
print(f"Version: {config.version}")
print(f"Description: {config.description}")

# Access workspace settings
settings = config.workspace_settings
print(f"Python version: {settings.python_version}")
print(f"Dependencies: {settings.dependencies}")
print(f"Environment variables: {settings.environment_variables}")
```

### Creating Workspace Configuration

```python
from cursus.validation.workspace.models import WorkspaceConfig, WorkspaceSettings

# Create workspace settings
settings = WorkspaceSettings(
    python_version="3.9",
    dependencies=["numpy", "pandas", "scikit-learn"],
    environment_variables={"MODEL_PATH": "/models", "DATA_PATH": "/data"},
    custom_paths=["/custom/lib"],
    validation_rules={
        "strict_typing": True,
        "require_docstrings": True,
        "max_complexity": 10
    }
)

# Create workspace configuration
config = WorkspaceConfig(
    workspace_name="ML Development Workspace",
    developer_id="developer_1",
    version="1.0.0",
    description="Machine learning model development workspace",
    workspace_settings=settings,
    created_at="2024-01-01T00:00:00Z",
    updated_at="2024-01-01T00:00:00Z"
)

# Save configuration
manager.save_workspace_config("developer_1", config)
print("Workspace configuration saved")
```

## Advanced Configuration Management

### Configuration Templates

```python
def create_ml_workspace_template():
    """Create a template for ML workspaces"""
    
    ml_settings = WorkspaceSettings(
        python_version="3.9",
        dependencies=[
            "numpy>=1.21.0",
            "pandas>=1.3.0",
            "scikit-learn>=1.0.0",
            "xgboost>=1.5.0",
            "boto3>=1.20.0"
        ],
        environment_variables={
            "MODEL_PATH": "/opt/ml/model",
            "DATA_PATH": "/opt/ml/input/data",
            "OUTPUT_PATH": "/opt/ml/output"
        },
        custom_paths=["/opt/ml/code"],
        validation_rules={
            "strict_typing": True,
            "require_docstrings": True,
            "max_complexity": 15,
            "min_test_coverage": 80
        }
    )
    
    return WorkspaceConfig(
        workspace_name="ML Template",
        developer_id="template",
        version="1.0.0",
        description="Standard ML development workspace template",
        workspace_settings=ml_settings
    )

def create_data_processing_template():
    """Create a template for data processing workspaces"""
    
    dp_settings = WorkspaceSettings(
        python_version="3.9",
        dependencies=[
            "pandas>=1.3.0",
            "numpy>=1.21.0",
            "pyarrow>=6.0.0",
            "dask>=2021.10.0"
        ],
        environment_variables={
            "DATA_SOURCE": "s3://data-bucket",
            "PROCESSING_MODE": "distributed"
        },
        validation_rules={
            "require_type_hints": True,
            "max_memory_usage": "8GB"
        }
    )
    
    return WorkspaceConfig(
        workspace_name="Data Processing Template",
        developer_id="template",
        version="1.0.0",
        description="Data processing workspace template",
        workspace_settings=dp_settings
    )

# Usage
ml_template = create_ml_workspace_template()
dp_template = create_data_processing_template()
```

### Configuration Inheritance

```python
def create_workspace_from_template(manager, template_config, new_developer_id, customizations=None):
    """Create a new workspace configuration from a template"""
    
    # Start with template settings
    new_settings = WorkspaceSettings(
        python_version=template_config.workspace_settings.python_version,
        dependencies=template_config.workspace_settings.dependencies.copy(),
        environment_variables=template_config.workspace_settings.environment_variables.copy(),
        custom_paths=template_config.workspace_settings.custom_paths.copy(),
        validation_rules=template_config.workspace_settings.validation_rules.copy()
    )
    
    # Apply customizations
    if customizations:
        if "dependencies" in customizations:
            new_settings.dependencies.extend(customizations["dependencies"])
        
        if "environment_variables" in customizations:
            new_settings.environment_variables.update(customizations["environment_variables"])
        
        if "validation_rules" in customizations:
            new_settings.validation_rules.update(customizations["validation_rules"])
    
    # Create new configuration
    new_config = WorkspaceConfig(
        workspace_name=f"{template_config.workspace_name} - {new_developer_id}",
        developer_id=new_developer_id,
        version="1.0.0",
        description=f"Customized workspace for {new_developer_id}",
        workspace_settings=new_settings
    )
    
    # Save the new configuration
    manager.save_workspace_config(new_developer_id, new_config)
    return new_config

# Usage
customizations = {
    "dependencies": ["tensorflow>=2.8.0"],
    "environment_variables": {"GPU_ENABLED": "true"},
    "validation_rules": {"require_gpu_tests": True}
}

new_config = create_workspace_from_template(
    manager, 
    ml_template, 
    "developer_2", 
    customizations
)
```

## Configuration Validation

### Validating Configuration Files

```python
def validate_workspace_configuration(config):
    """Validate workspace configuration for completeness and correctness"""
    
    validation_errors = []
    
    # Check required fields
    if not config.workspace_name:
        validation_errors.append("Workspace name is required")
    
    if not config.developer_id:
        validation_errors.append("Developer ID is required")
    
    # Validate settings
    settings = config.workspace_settings
    
    # Check Python version format
    import re
    if not re.match(r'^\d+\.\d+$', settings.python_version):
        validation_errors.append(f"Invalid Python version format: {settings.python_version}")
    
    # Check dependencies format
    for dep in settings.dependencies:
        if not isinstance(dep, str):
            validation_errors.append(f"Invalid dependency format: {dep}")
    
    # Check environment variables
    for key, value in settings.environment_variables.items():
        if not isinstance(key, str) or not isinstance(value, str):
            validation_errors.append(f"Invalid environment variable: {key}={value}")
    
    # Check custom paths exist
    import os
    for path in settings.custom_paths:
        if not os.path.exists(path):
            validation_errors.append(f"Custom path does not exist: {path}")
    
    return validation_errors

# Usage
config = manager.load_workspace_config("developer_1")
errors = validate_workspace_configuration(config)

if errors:
    print("Configuration validation errors:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Configuration is valid")
```

### Configuration Schema Validation

```python
from pydantic import ValidationError

def validate_config_schema(config_dict):
    """Validate configuration against Pydantic schema"""
    
    try:
        # This will raise ValidationError if invalid
        config = WorkspaceConfig(**config_dict)
        return True, config, []
    except ValidationError as e:
        errors = []
        for error in e.errors():
            field = " -> ".join(str(x) for x in error['loc'])
            message = error['msg']
            errors.append(f"{field}: {message}")
        return False, None, errors

# Usage
config_data = {
    "workspace_name": "Test Workspace",
    "developer_id": "test_dev",
    "version": "1.0.0",
    "workspace_settings": {
        "python_version": "3.9",
        "dependencies": ["numpy"],
        "environment_variables": {},
        "custom_paths": [],
        "validation_rules": {}
    }
}

is_valid, config, errors = validate_config_schema(config_data)
if not is_valid:
    print("Schema validation errors:")
    for error in errors:
        print(f"  - {error}")
```

## Configuration Migration

### Version Migration

```python
def migrate_config_v1_to_v2(old_config_dict):
    """Migrate configuration from version 1.0 to 2.0"""
    
    # Version 2.0 changes:
    # - Added validation_rules
    # - Renamed some fields
    # - Added new required fields
    
    migrated = old_config_dict.copy()
    
    # Add new fields with defaults
    if "validation_rules" not in migrated.get("workspace_settings", {}):
        migrated["workspace_settings"]["validation_rules"] = {
            "strict_typing": False,
            "require_docstrings": False
        }
    
    # Rename fields
    if "env_vars" in migrated.get("workspace_settings", {}):
        migrated["workspace_settings"]["environment_variables"] = \
            migrated["workspace_settings"].pop("env_vars")
    
    # Update version
    migrated["version"] = "2.0.0"
    
    return migrated

def auto_migrate_workspace_config(manager, developer_id):
    """Automatically migrate workspace configuration to latest version"""
    
    try:
        config = manager.load_workspace_config(developer_id)
        current_version = config.version
        
        if current_version == "1.0.0":
            print(f"Migrating {developer_id} from v1.0.0 to v2.0.0")
            
            # Convert to dict, migrate, then back to object
            config_dict = config.dict()
            migrated_dict = migrate_config_v1_to_v2(config_dict)
            
            # Validate migrated configuration
            is_valid, migrated_config, errors = validate_config_schema(migrated_dict)
            
            if is_valid:
                manager.save_workspace_config(developer_id, migrated_config)
                print(f"Migration successful for {developer_id}")
                return migrated_config
            else:
                print(f"Migration failed for {developer_id}: {errors}")
                return None
        else:
            print(f"{developer_id} is already at latest version")
            return config
            
    except Exception as e:
        print(f"Migration failed for {developer_id}: {e}")
        return None

# Usage
migrated_config = auto_migrate_workspace_config(manager, "developer_1")
```

## Configuration Synchronization

### Multi-Environment Configuration

```python
class MultiEnvironmentConfigManager:
    """Manage configurations across multiple environments"""
    
    def __init__(self, workspace_root):
        self.managers = {
            "development": WorkspaceManager(f"{workspace_root}/dev"),
            "staging": WorkspaceManager(f"{workspace_root}/staging"),
            "production": WorkspaceManager(f"{workspace_root}/prod")
        }
    
    def sync_config_across_environments(self, developer_id, source_env="development"):
        """Sync configuration from source environment to others"""
        
        # Load source configuration
        source_config = self.managers[source_env].load_workspace_config(developer_id)
        
        results = {}
        
        for env_name, manager in self.managers.items():
            if env_name == source_env:
                continue
            
            try:
                # Customize config for target environment
                target_config = self._customize_for_environment(source_config, env_name)
                
                # Save to target environment
                manager.save_workspace_config(developer_id, target_config)
                results[env_name] = "success"
                
            except Exception as e:
                results[env_name] = f"failed: {e}"
        
        return results
    
    def _customize_for_environment(self, config, environment):
        """Customize configuration for specific environment"""
        
        # Create a copy
        env_config = config.copy(deep=True)
        
        # Environment-specific customizations
        if environment == "production":
            # Production settings
            env_config.workspace_settings.validation_rules.update({
                "strict_typing": True,
                "require_docstrings": True,
                "min_test_coverage": 90
            })
            env_config.workspace_settings.environment_variables.update({
                "LOG_LEVEL": "WARNING",
                "DEBUG": "false"
            })
        elif environment == "staging":
            # Staging settings
            env_config.workspace_settings.environment_variables.update({
                "LOG_LEVEL": "INFO",
                "DEBUG": "false"
            })
        
        return env_config

# Usage
multi_manager = MultiEnvironmentConfigManager("/path/to/workspaces")
sync_results = multi_manager.sync_config_across_environments("developer_1")

for env, result in sync_results.items():
    print(f"{env}: {result}")
```

### Configuration Backup and Restore

```python
import json
import datetime
from pathlib import Path

class ConfigurationBackupManager:
    """Manage configuration backups"""
    
    def __init__(self, backup_root):
        self.backup_root = Path(backup_root)
        self.backup_root.mkdir(parents=True, exist_ok=True)
    
    def backup_workspace_config(self, manager, developer_id):
        """Create a backup of workspace configuration"""
        
        config = manager.load_workspace_config(developer_id)
        
        # Create backup filename with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_root / f"{developer_id}_config_{timestamp}.json"
        
        # Save backup
        with open(backup_file, 'w') as f:
            json.dump(config.dict(), f, indent=2, default=str)
        
        print(f"Configuration backed up to {backup_file}")
        return backup_file
    
    def restore_workspace_config(self, manager, developer_id, backup_file):
        """Restore workspace configuration from backup"""
        
        with open(backup_file, 'r') as f:
            config_dict = json.load(f)
        
        # Validate and create config object
        is_valid, config, errors = validate_config_schema(config_dict)
        
        if is_valid:
            manager.save_workspace_config(developer_id, config)
            print(f"Configuration restored for {developer_id}")
            return config
        else:
            print(f"Restore failed: {errors}")
            return None
    
    def list_backups(self, developer_id=None):
        """List available backups"""
        
        pattern = f"{developer_id}_config_*.json" if developer_id else "*_config_*.json"
        backups = list(self.backup_root.glob(pattern))
        
        backup_info = []
        for backup in sorted(backups):
            # Extract timestamp from filename
            parts = backup.stem.split('_')
            if len(parts) >= 3:
                dev_id = parts[0]
                timestamp = parts[2]
                backup_info.append({
                    'developer_id': dev_id,
                    'timestamp': timestamp,
                    'file': backup
                })
        
        return backup_info

# Usage
backup_manager = ConfigurationBackupManager("/path/to/backups")

# Create backup
backup_file = backup_manager.backup_workspace_config(manager, "developer_1")

# List backups
backups = backup_manager.list_backups("developer_1")
for backup in backups:
    print(f"{backup['developer_id']}: {backup['timestamp']}")

# Restore from backup
if backups:
    latest_backup = backups[-1]['file']
    restored_config = backup_manager.restore_workspace_config(
        manager, "developer_1", latest_backup
    )
```

## Best Practices

### 1. Configuration Validation Pipeline

```python
def comprehensive_config_validation(manager, developer_id):
    """Comprehensive configuration validation pipeline"""
    
    validation_steps = [
        ("Schema Validation", validate_config_schema),
        ("Business Rules", validate_workspace_configuration),
        ("Environment Check", validate_environment_compatibility),
        ("Dependency Check", validate_dependencies)
    ]
    
    config = manager.load_workspace_config(developer_id)
    config_dict = config.dict()
    
    all_passed = True
    results = {}
    
    for step_name, validator in validation_steps:
        try:
            if step_name == "Schema Validation":
                is_valid, _, errors = validator(config_dict)
            else:
                errors = validator(config)
                is_valid = len(errors) == 0
            
            results[step_name] = {
                'passed': is_valid,
                'errors': errors
            }
            
            if not is_valid:
                all_passed = False
                
        except Exception as e:
            results[step_name] = {
                'passed': False,
                'errors': [f"Validation error: {e}"]
            }
            all_passed = False
    
    return all_passed, results

def validate_environment_compatibility(config):
    """Validate environment compatibility"""
    errors = []
    
    # Check Python version compatibility
    import sys
    current_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    required_python = config.workspace_settings.python_version
    
    if current_python != required_python:
        errors.append(f"Python version mismatch: required {required_python}, current {current_python}")
    
    return errors

def validate_dependencies(config):
    """Validate dependencies are available"""
    errors = []
    
    for dep in config.workspace_settings.dependencies:
        try:
            # Simple check - try to import
            import importlib
            package_name = dep.split('>=')[0].split('==')[0]
            importlib.import_module(package_name)
        except ImportError:
            errors.append(f"Dependency not available: {dep}")
    
    return errors

# Usage
passed, validation_results = comprehensive_config_validation(manager, "developer_1")

print(f"Overall validation: {'PASSED' if passed else 'FAILED'}")
for step, result in validation_results.items():
    status = "PASS" if result['passed'] else "FAIL"
    print(f"  {step}: {status}")
    if result['errors']:
        for error in result['errors']:
            print(f"    - {error}")
```

### 2. Configuration Monitoring

```python
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ConfigurationMonitor(FileSystemEventHandler):
    """Monitor configuration file changes"""
    
    def __init__(self, manager, callback=None):
        self.manager = manager
        self.callback = callback
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        if event.src_path.endswith('workspace_config.json'):
            print(f"Configuration file changed: {event.src_path}")
            
            # Extract developer ID from path
            developer_id = Path(event.src_path).parent.name
            
            try:
                # Reload and validate configuration
                config = self.manager.load_workspace_config(developer_id)
                errors = validate_workspace_configuration(config)
                
                if errors:
                    print(f"Configuration validation failed for {developer_id}:")
                    for error in errors:
                        print(f"  - {error}")
                else:
                    print(f"Configuration validated successfully for {developer_id}")
                
                # Call callback if provided
                if self.callback:
                    self.callback(developer_id, config, errors)
                    
            except Exception as e:
                print(f"Failed to process configuration change: {e}")

def start_configuration_monitoring(manager, workspace_root):
    """Start monitoring configuration changes"""
    
    event_handler = ConfigurationMonitor(manager)
    observer = Observer()
    observer.schedule(event_handler, workspace_root, recursive=True)
    observer.start()
    
    print(f"Started configuration monitoring for {workspace_root}")
    return observer

# Usage
# observer = start_configuration_monitoring(manager, "/path/to/workspaces")
# Keep the observer running...
# observer.stop()
# observer.join()
```

### 3. Configuration Templates Library

```python
class ConfigurationTemplateLibrary:
    """Library of configuration templates"""
    
    def __init__(self):
        self.templates = {}
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default configuration templates"""
        
        # Machine Learning Template
        self.templates["ml_basic"] = create_ml_workspace_template()
        
        # Data Processing Template
        self.templates["data_processing"] = create_data_processing_template()
        
        # Web Development Template
        self.templates["web_dev"] = self._create_web_dev_template()
    
    def _create_web_dev_template(self):
        """Create web development template"""
        
        web_settings = WorkspaceSettings(
            python_version="3.9",
            dependencies=[
                "flask>=2.0.0",
                "django>=4.0.0",
                "fastapi>=0.70.0",
                "requests>=2.25.0"
            ],
            environment_variables={
                "FLASK_ENV": "development",
                "DEBUG": "true"
            },
            validation_rules={
                "require_tests": True,
                "min_test_coverage": 70
            }
        )
        
        return WorkspaceConfig(
            workspace_name="Web Development Template",
            developer_id="template",
            version="1.0.0",
            description="Web development workspace template",
            workspace_settings=web_settings
        )
    
    def get_template(self, template_name):
        """Get a configuration template"""
        return self.templates.get(template_name)
    
    def list_templates(self):
        """List available templates"""
        return list(self.templates.keys())
    
    def add_custom_template(self, name, template_config):
        """Add a custom template"""
        self.templates[name] = template_config

# Usage
template_library = ConfigurationTemplateLibrary()

print("Available templates:")
for template_name in template_library.list_templates():
    template = template_library.get_template(template_name)
    print(f"  {template_name}: {template.description}")

# Use a template
ml_template = template_library.get_template("ml_basic")
new_config = create_workspace_from_template(
    manager, ml_template, "new_developer"
)

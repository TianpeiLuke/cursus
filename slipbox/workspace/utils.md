---
tags:
  - code
  - workspace
  - utilities
  - helpers
  - configuration
keywords:
  - WorkspaceConfig
  - PathUtils
  - ConfigUtils
  - FileUtils
  - ValidationUtils
  - TimeUtils
  - LoggingUtils
  - WorkspaceUtils
topics:
  - workspace utilities
  - path management
  - configuration helpers
  - file operations
language: python
date of note: 2024-12-07
---

# Workspace Utilities

Comprehensive utility functions for workspace operations including path management, configuration helpers, file operations, validation utilities, and common workspace tasks.

## Overview

The Workspace Utilities module provides a comprehensive set of utility functions and classes for workspace operations. It includes path management utilities, configuration helpers, file operations, validation utilities, time-based operations, logging utilities, and high-level workspace management functions.

The module implements type-safe utility classes with Pydantic models for configuration management, comprehensive path safety checks, flexible configuration loading and merging, file operations with backup capabilities, and validation utilities for workspace compliance checking.

## Classes and Methods

### Configuration Classes
- [`WorkspaceConfig`](#workspaceconfig) - Workspace configuration model with validation
- [`ConfigUtils`](#configutils) - Configuration management utilities

### Utility Classes
- [`PathUtils`](#pathutils) - Path operations and safety utilities
- [`FileUtils`](#fileutils) - File operations and management utilities
- [`ValidationUtils`](#validationutils) - Validation operations for workspace compliance
- [`TimeUtils`](#timeutils) - Time-based operations and age checking
- [`LoggingUtils`](#loggingutils) - Logging setup and structured logging
- [`WorkspaceUtils`](#workspaceutils) - High-level workspace utility functions

## API Reference

### WorkspaceConfig

_class_ cursus.workspace.utils.WorkspaceConfig(_workspace_id_, _base_path_, _isolation_mode="strict"_, _auto_cleanup=True_, _cleanup_threshold_days=30_, _max_workspace_size_mb=None_, _allowed_extensions=[]_, _excluded_patterns=[]_)

Workspace configuration model with validation and type safety for workspace settings.

**Parameters:**
- **workspace_id** (_str_) – Workspace identifier (minimum length 1)
- **base_path** (_Path_) – Base workspace path
- **isolation_mode** (_str_) – Isolation mode (default "strict")
- **auto_cleanup** (_bool_) – Enable automatic cleanup (default True)
- **cleanup_threshold_days** (_int_) – Days before cleanup (minimum 1, default 30)
- **max_workspace_size_mb** (_Optional[int]_) – Maximum workspace size in MB
- **allowed_extensions** (_List[str]_) – Allowed file extensions
- **excluded_patterns** (_List[str]_) – Patterns to exclude from workspace

```python
from cursus.workspace.utils import WorkspaceConfig
from pathlib import Path

# Create workspace configuration
config = WorkspaceConfig(
    workspace_id="alice_workspace",
    base_path=Path("/workspaces/alice"),
    isolation_mode="strict",
    auto_cleanup=True,
    cleanup_threshold_days=30,
    max_workspace_size_mb=1000,
    allowed_extensions=[".py", ".yaml", ".json", ".md"],
    excluded_patterns=["__pycache__", "*.pyc", ".git"]
)

print(f"Workspace ID: {config.workspace_id}")
print(f"Base path: {config.base_path}")
print(f"Auto cleanup: {config.auto_cleanup}")
```

### PathUtils

_class_ cursus.workspace.utils.PathUtils()

Utilities for path operations including normalization, safety checks, and directory management.

```python
from cursus.workspace.utils import PathUtils
from pathlib import Path

# Path utilities
path_utils = PathUtils()
```

#### normalize_workspace_path

normalize_workspace_path(_path_)

Normalize workspace path to absolute path with proper resolution.

**Parameters:**
- **path** (_Union[str, Path]_) – Path to normalize

**Returns:**
- **Path** – Normalized absolute path

```python
# Normalize workspace path
normalized = PathUtils.normalize_workspace_path("./workspace")
print(f"Normalized path: {normalized}")

# Works with Path objects too
path_obj = Path("../workspaces/alice")
normalized = PathUtils.normalize_workspace_path(path_obj)
```

#### is_safe_path

is_safe_path(_path_, _base_path_)

Check if path is safe by ensuring it stays within the base path boundaries.

**Parameters:**
- **path** (_Union[str, Path]_) – Path to check for safety
- **base_path** (_Union[str, Path]_) – Base path for safety boundary

**Returns:**
- **bool** – True if path is safe (within base path), False otherwise

```python
# Check path safety
base_path = "/workspaces"
safe_path = "/workspaces/alice/script.py"
unsafe_path = "/etc/passwd"

is_safe = PathUtils.is_safe_path(safe_path, base_path)
print(f"Safe path: {is_safe}")  # True

is_safe = PathUtils.is_safe_path(unsafe_path, base_path)
print(f"Unsafe path: {is_safe}")  # False
```

#### get_relative_path

get_relative_path(_path_, _base_path_)

Get relative path from base path if possible.

**Parameters:**
- **path** (_Union[str, Path]_) – Target path
- **base_path** (_Union[str, Path]_) – Base path for relative calculation

**Returns:**
- **Optional[Path]** – Relative path if possible, None otherwise

```python
# Get relative path
base_path = "/workspaces"
target_path = "/workspaces/alice/scripts/train.py"

relative = PathUtils.get_relative_path(target_path, base_path)
print(f"Relative path: {relative}")  # alice/scripts/train.py
```

#### ensure_directory

ensure_directory(_path_)

Ensure directory exists, creating it and parent directories if necessary.

**Parameters:**
- **path** (_Union[str, Path]_) – Directory path to ensure

**Returns:**
- **bool** – True if successful, False otherwise

```python
# Ensure directory exists
success = PathUtils.ensure_directory("/workspaces/new_workspace/scripts")
if success:
    print("Directory created successfully")
```

#### get_directory_size

get_directory_size(_path_)

Get total size of directory and all its contents in bytes.

**Parameters:**
- **path** (_Union[str, Path]_) – Directory path to measure

**Returns:**
- **int** – Total size in bytes

```python
# Get directory size
size_bytes = PathUtils.get_directory_size("/workspaces/alice")
size_mb = size_bytes / (1024 * 1024)
print(f"Workspace size: {size_mb:.2f} MB")
```

#### clean_path_patterns

clean_path_patterns(_path_, _patterns_)

Clean files and directories matching specified patterns.

**Parameters:**
- **path** (_Union[str, Path]_) – Base path to clean
- **patterns** (_List[str]_) – Patterns to match for cleanup

**Returns:**
- **int** – Number of items cleaned

```python
# Clean temporary files
patterns = ["__pycache__", "*.pyc", "*.tmp", ".DS_Store"]
cleaned_count = PathUtils.clean_path_patterns("/workspaces/alice", patterns)
print(f"Cleaned {cleaned_count} items")
```

### ConfigUtils

_class_ cursus.workspace.utils.ConfigUtils()

Utilities for configuration management including loading, saving, merging, and validation.

```python
from cursus.workspace.utils import ConfigUtils
```

#### load_config

load_config(_config_path_)

Load configuration from YAML or JSON file.

**Parameters:**
- **config_path** (_Union[str, Path]_) – Path to configuration file

**Returns:**
- **Optional[Dict[str, Any]]** – Configuration dictionary or None if failed

```python
# Load configuration
config = ConfigUtils.load_config("/workspaces/alice/.workspace_config.yaml")
if config:
    print(f"Loaded config: {config}")
else:
    print("Failed to load configuration")
```

#### save_config

save_config(_config_, _config_path_)

Save configuration dictionary to YAML or JSON file.

**Parameters:**
- **config** (_Dict[str, Any]_) – Configuration dictionary to save
- **config_path** (_Union[str, Path]_) – Path to save configuration file

**Returns:**
- **bool** – True if successful, False otherwise

```python
# Save configuration
config = {
    "workspace_id": "alice",
    "settings": {
        "auto_cleanup": True,
        "max_size_mb": 1000
    }
}

success = ConfigUtils.save_config(config, "/workspaces/alice/config.yaml")
if success:
    print("Configuration saved successfully")
```

#### merge_configs

merge_configs(_base_config_, _override_config_)

Merge configuration dictionaries with deep merging for nested dictionaries.

**Parameters:**
- **base_config** (_Dict[str, Any]_) – Base configuration dictionary
- **override_config** (_Dict[str, Any]_) – Override configuration dictionary

**Returns:**
- **Dict[str, Any]** – Merged configuration dictionary

```python
# Merge configurations
base_config = {
    "workspace_id": "alice",
    "settings": {
        "auto_cleanup": True,
        "cleanup_days": 30
    }
}

override_config = {
    "settings": {
        "cleanup_days": 60,
        "max_size_mb": 2000
    }
}

merged = ConfigUtils.merge_configs(base_config, override_config)
print(f"Merged config: {merged}")
# Result: workspace_id="alice", settings={auto_cleanup=True, cleanup_days=60, max_size_mb=2000}
```

#### validate_config

validate_config(_config_, _schema_)

Validate configuration dictionary against a schema.

**Parameters:**
- **config** (_Dict[str, Any]_) – Configuration to validate
- **schema** (_Dict[str, Any]_) – Validation schema

**Returns:**
- **Tuple[bool, List[str]]** – Tuple of (is_valid, error_messages)

```python
# Validate configuration
config = {"workspace_id": "alice", "max_size": 1000}
schema = {
    "workspace_id": {"required": True, "type": str},
    "max_size": {"required": False, "type": int}
}

is_valid, errors = ConfigUtils.validate_config(config, schema)
if is_valid:
    print("Configuration is valid")
else:
    print(f"Validation errors: {errors}")
```

### FileUtils

_class_ cursus.workspace.utils.FileUtils()

Utilities for file operations including hashing, backup, and file discovery.

```python
from cursus.workspace.utils import FileUtils
```

#### calculate_file_hash

calculate_file_hash(_file_path_, _algorithm="sha256"_)

Calculate hash of file using specified algorithm.

**Parameters:**
- **file_path** (_Union[str, Path]_) – Path to file
- **algorithm** (_str_) – Hash algorithm to use (default "sha256")

**Returns:**
- **Optional[str]** – Hash string or None if failed

```python
# Calculate file hash
hash_value = FileUtils.calculate_file_hash("/workspaces/alice/script.py")
if hash_value:
    print(f"File hash: {hash_value}")

# Use different algorithm
md5_hash = FileUtils.calculate_file_hash("/workspaces/alice/script.py", "md5")
```

#### is_text_file

is_text_file(_file_path_)

Check if file is a text file by examining its content.

**Parameters:**
- **file_path** (_Union[str, Path]_) – Path to file

**Returns:**
- **bool** – True if text file, False otherwise

```python
# Check if file is text
is_text = FileUtils.is_text_file("/workspaces/alice/script.py")
print(f"Is text file: {is_text}")  # True

is_text = FileUtils.is_text_file("/workspaces/alice/model.pkl")
print(f"Is text file: {is_text}")  # False
```

#### backup_file

backup_file(_file_path_, _backup_dir=None_)

Create backup of file with timestamp.

**Parameters:**
- **file_path** (_Union[str, Path]_) – Path to file to backup
- **backup_dir** (_Optional[Union[str, Path]]_) – Directory for backup (default: same directory)

**Returns:**
- **Optional[Path]** – Path to backup file or None if failed

```python
# Create backup in same directory
backup_path = FileUtils.backup_file("/workspaces/alice/config.yaml")
if backup_path:
    print(f"Backup created: {backup_path}")

# Create backup in specific directory
backup_path = FileUtils.backup_file(
    "/workspaces/alice/config.yaml",
    "/backups"
)
```

#### find_files

find_files(_directory_, _pattern="*"_, _recursive=True_)

Find files matching pattern in directory.

**Parameters:**
- **directory** (_Union[str, Path]_) – Directory to search
- **pattern** (_str_) – File pattern to match (default "*")
- **recursive** (_bool_) – Whether to search recursively (default True)

**Returns:**
- **List[Path]** – List of matching file paths

```python
# Find all Python files
python_files = FileUtils.find_files("/workspaces/alice", "*.py")
print(f"Found {len(python_files)} Python files")

# Find YAML files in specific directory only
yaml_files = FileUtils.find_files("/workspaces/alice", "*.yaml", recursive=False)
```

### ValidationUtils

_class_ cursus.workspace.utils.ValidationUtils()

Utilities for validation operations including workspace structure and compliance checking.

```python
from cursus.workspace.utils import ValidationUtils
```

#### validate_workspace_structure

validate_workspace_structure(_workspace_path_, _required_dirs_)

Validate workspace directory structure against required directories.

**Parameters:**
- **workspace_path** (_Union[str, Path]_) – Path to workspace
- **required_dirs** (_List[str]_) – List of required directories

**Returns:**
- **Tuple[bool, List[str]]** – Tuple of (is_valid, missing_directories)

```python
# Validate workspace structure
required_dirs = ["builders", "configs", "contracts", "specs", "scripts"]
is_valid, missing = ValidationUtils.validate_workspace_structure(
    "/workspaces/alice", 
    required_dirs
)

if is_valid:
    print("Workspace structure is valid")
else:
    print(f"Missing directories: {missing}")
```

#### validate_file_extensions

validate_file_extensions(_directory_, _allowed_extensions_)

Validate file extensions in directory against allowed extensions.

**Parameters:**
- **directory** (_Union[str, Path]_) – Directory to validate
- **allowed_extensions** (_List[str]_) – List of allowed file extensions

**Returns:**
- **Tuple[bool, List[Path]]** – Tuple of (is_valid, invalid_files)

```python
# Validate file extensions
allowed_extensions = [".py", ".yaml", ".json", ".md", ".txt"]
is_valid, invalid_files = ValidationUtils.validate_file_extensions(
    "/workspaces/alice",
    allowed_extensions
)

if not is_valid:
    print(f"Invalid files found: {len(invalid_files)}")
    for file_path in invalid_files[:5]:  # Show first 5
        print(f"  - {file_path}")
```

#### validate_workspace_size

validate_workspace_size(_workspace_path_, _max_size_mb_)

Validate workspace size against maximum allowed size.

**Parameters:**
- **workspace_path** (_Union[str, Path]_) – Path to workspace
- **max_size_mb** (_int_) – Maximum size in MB

**Returns:**
- **Tuple[bool, int]** – Tuple of (is_valid, current_size_mb)

```python
# Validate workspace size
is_valid, current_size = ValidationUtils.validate_workspace_size(
    "/workspaces/alice", 
    1000  # 1GB limit
)

print(f"Current size: {current_size} MB")
if not is_valid:
    print("Workspace exceeds size limit")
```

### TimeUtils

_class_ cursus.workspace.utils.TimeUtils()

Utilities for time-based operations including age checking and timestamp formatting.

```python
from cursus.workspace.utils import TimeUtils
```

#### is_path_older_than

is_path_older_than(_path_, _days_)

Check if path is older than specified number of days.

**Parameters:**
- **path** (_Union[str, Path]_) – Path to check
- **days** (_int_) – Number of days

**Returns:**
- **bool** – True if older than specified days, False otherwise

```python
# Check if workspace is old
is_old = TimeUtils.is_path_older_than("/workspaces/alice", 30)
if is_old:
    print("Workspace is older than 30 days")
```

#### get_path_age_days

get_path_age_days(_path_)

Get age of path in days since last modification.

**Parameters:**
- **path** (_Union[str, Path]_) – Path to check

**Returns:**
- **Optional[int]** – Age in days or None if failed

```python
# Get workspace age
age_days = TimeUtils.get_path_age_days("/workspaces/alice")
if age_days is not None:
    print(f"Workspace age: {age_days} days")
```

#### format_timestamp

format_timestamp(_timestamp=None_, _format_str="%Y-%m-%d %H:%M:%S"_)

Format timestamp to string with specified format.

**Parameters:**
- **timestamp** (_Optional[datetime]_) – Timestamp to format (default: now)
- **format_str** (_str_) – Format string (default: "%Y-%m-%d %H:%M:%S")

**Returns:**
- **str** – Formatted timestamp string

```python
from datetime import datetime

# Format current time
formatted = TimeUtils.format_timestamp()
print(f"Current time: {formatted}")

# Format specific timestamp
timestamp = datetime(2024, 12, 7, 15, 30, 0)
formatted = TimeUtils.format_timestamp(timestamp, "%Y-%m-%d")
print(f"Date: {formatted}")  # 2024-12-07
```

### LoggingUtils

_class_ cursus.workspace.utils.LoggingUtils()

Utilities for logging operations including logger setup and structured logging.

```python
from cursus.workspace.utils import LoggingUtils
```

#### setup_workspace_logger

setup_workspace_logger(_workspace_id_, _log_level="INFO"_)

Set up logger for workspace operations with proper formatting.

**Parameters:**
- **workspace_id** (_str_) – Workspace identifier
- **log_level** (_str_) – Logging level (default "INFO")

**Returns:**
- **logging.Logger** – Configured logger instance

```python
# Set up workspace logger
logger = LoggingUtils.setup_workspace_logger("alice_workspace", "DEBUG")

# Use logger
logger.info("Workspace operation started")
logger.debug("Debug information")
logger.warning("Warning message")
```

#### log_workspace_operation

log_workspace_operation(_logger_, _operation_, _details_)

Log workspace operation with structured details.

**Parameters:**
- **logger** (_logging.Logger_) – Logger instance
- **operation** (_str_) – Operation name
- **details** (_Dict[str, Any]_) – Operation details

```python
# Log structured operation
logger = LoggingUtils.setup_workspace_logger("alice_workspace")

operation_details = {
    "workspace_path": "/workspaces/alice",
    "files_processed": 15,
    "duration_seconds": 2.5
}

LoggingUtils.log_workspace_operation(
    logger, 
    "workspace_cleanup", 
    operation_details
)
```

### WorkspaceUtils

_class_ cursus.workspace.utils.WorkspaceUtils()

High-level workspace utility functions for common workspace operations.

```python
from cursus.workspace.utils import WorkspaceUtils
```

#### create_workspace_config

create_workspace_config(_workspace_id_, _base_path_, _**kwargs_)

Create workspace configuration with default settings.

**Parameters:**
- **workspace_id** (_str_) – Workspace identifier
- **base_path** (_Union[str, Path]_) – Base workspace path
- **kwargs** – Additional configuration options

**Returns:**
- **WorkspaceConfig** – WorkspaceConfig instance

```python
# Create workspace configuration
config = WorkspaceUtils.create_workspace_config(
    "alice_workspace",
    "/workspaces/alice",
    max_workspace_size_mb=2000,
    cleanup_threshold_days=60
)

print(f"Created config for: {config.workspace_id}")
```

#### initialize_workspace_directory

initialize_workspace_directory(_workspace_path_, _config_)

Initialize workspace directory structure with standard directories and configuration.

**Parameters:**
- **workspace_path** (_Union[str, Path]_) – Path to workspace
- **config** (_WorkspaceConfig_) – Workspace configuration

**Returns:**
- **bool** – True if successful, False otherwise

```python
# Initialize workspace directory
config = WorkspaceUtils.create_workspace_config("alice", "/workspaces/alice")
success = WorkspaceUtils.initialize_workspace_directory("/workspaces/alice", config)

if success:
    print("Workspace directory initialized successfully")
```

#### cleanup_workspace

cleanup_workspace(_workspace_path_, _config_)

Clean up workspace according to configuration settings.

**Parameters:**
- **workspace_path** (_Union[str, Path]_) – Path to workspace
- **config** (_WorkspaceConfig_) – Workspace configuration

**Returns:**
- **Tuple[bool, int]** – Tuple of (success, items_cleaned)

```python
# Clean up workspace
config = WorkspaceUtils.create_workspace_config("alice", "/workspaces/alice")
success, items_cleaned = WorkspaceUtils.cleanup_workspace("/workspaces/alice", config)

print(f"Cleanup successful: {success}")
print(f"Items cleaned: {items_cleaned}")
```

#### validate_workspace

validate_workspace(_workspace_path_, _config_)

Validate workspace according to configuration requirements.

**Parameters:**
- **workspace_path** (_Union[str, Path]_) – Path to workspace
- **config** (_WorkspaceConfig_) – Workspace configuration

**Returns:**
- **Tuple[bool, List[str]]** – Tuple of (is_valid, validation_errors)

```python
# Validate workspace
config = WorkspaceUtils.create_workspace_config("alice", "/workspaces/alice")
is_valid, errors = WorkspaceUtils.validate_workspace("/workspaces/alice", config)

if is_valid:
    print("Workspace validation passed")
else:
    print("Validation errors:")
    for error in errors:
        print(f"  - {error}")
```

## Usage Examples

### Complete Workspace Setup and Management

```python
from cursus.workspace.utils import WorkspaceUtils, PathUtils, ConfigUtils
from pathlib import Path

# Complete workspace setup workflow
def setup_new_workspace(workspace_id, base_path):
    """Set up a new workspace with full configuration."""
    
    print(f"Setting up workspace: {workspace_id}")
    
    # 1. Create workspace configuration
    config = WorkspaceUtils.create_workspace_config(
        workspace_id=workspace_id,
        base_path=base_path,
        max_workspace_size_mb=1000,
        cleanup_threshold_days=30,
        allowed_extensions=[".py", ".yaml", ".json", ".md", ".txt"],
        excluded_patterns=["__pycache__", "*.pyc", ".git", ".DS_Store"]
    )
    
    # 2. Ensure base directory exists
    workspace_path = Path(base_path) / workspace_id
    if not PathUtils.ensure_directory(workspace_path):
        print("Failed to create workspace directory")
        return False
    
    # 3. Initialize workspace structure
    if not WorkspaceUtils.initialize_workspace_directory(workspace_path, config):
        print("Failed to initialize workspace structure")
        return False
    
    # 4. Validate workspace
    is_valid, errors = WorkspaceUtils.validate_workspace(workspace_path, config)
    if not is_valid:
        print("Workspace validation failed:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    # 5. Set up logging
    from cursus.workspace.utils import LoggingUtils
    logger = LoggingUtils.setup_workspace_logger(workspace_id)
    
    LoggingUtils.log_workspace_operation(
        logger,
        "workspace_setup",
        {
            "workspace_id": workspace_id,
            "workspace_path": str(workspace_path),
            "config": config.model_dump()
        }
    )
    
    print(f"✓ Workspace setup completed: {workspace_path}")
    return True

# Set up new workspace
setup_new_workspace("data_scientist_1", "/workspaces")
```

### Workspace Maintenance and Cleanup

```python
from cursus.workspace.utils import WorkspaceUtils, TimeUtils, PathUtils
import os

def maintain_workspaces(workspaces_root, max_age_days=60):
    """Maintain workspaces by cleaning up old and oversized ones."""
    
    workspaces_root = Path(workspaces_root)
    maintenance_report = {
        "workspaces_checked": 0,
        "workspaces_cleaned": 0,
        "old_workspaces": [],
        "oversized_workspaces": [],
        "errors": []
    }
    
    print("Starting workspace maintenance...")
    
    # Find all workspace directories
    for workspace_dir in workspaces_root.iterdir():
        if not workspace_dir.is_dir():
            continue
        
        maintenance_report["workspaces_checked"] += 1
        workspace_id = workspace_dir.name
        
        print(f"\nChecking workspace: {workspace_id}")
        
        try:
            # Load workspace configuration
            config_path = workspace_dir / ".workspace_config.yaml"
            if config_path.exists():
                config_data = ConfigUtils.load_config(config_path)
                if config_data:
                    config = WorkspaceConfig(**config_data)
                else:
                    print(f"  ⚠ Failed to load config for {workspace_id}")
                    continue
            else:
                # Create default config
                config = WorkspaceUtils.create_workspace_config(workspace_id, workspace_dir)
            
            # Check workspace age
            age_days = TimeUtils.get_path_age_days(workspace_dir)
            if age_days and age_days > max_age_days:
                print(f"  📅 Workspace is {age_days} days old (threshold: {max_age_days})")
                maintenance_report["old_workspaces"].append({
                    "workspace_id": workspace_id,
                    "age_days": age_days
                })
            
            # Check workspace size
            if config.max_workspace_size_mb:
                is_size_valid, current_size_mb = ValidationUtils.validate_workspace_size(
                    workspace_dir, 
                    config.max_workspace_size_mb
                )
                
                if not is_size_valid:
                    print(f"  💾 Workspace size: {current_size_mb}MB (limit: {config.max_workspace_size_mb}MB)")
                    maintenance_report["oversized_workspaces"].append({
                        "workspace_id": workspace_id,
                        "size_mb": current_size_mb,
                        "limit_mb": config.max_workspace_size_mb
                    })
            
            # Perform cleanup if auto_cleanup is enabled
            if config.auto_cleanup:
                success, items_cleaned = WorkspaceUtils.cleanup_workspace(workspace_dir, config)
                if success and items_cleaned > 0:
                    print(f"  🧹 Cleaned {items_cleaned} items")
                    maintenance_report["workspaces_cleaned"] += 1
            
        except Exception as e:
            error_msg = f"Error processing {workspace_id}: {e}"
            print(f"  ❌ {error_msg}")
            maintenance_report["errors"].append(error_msg)
    
    # Print maintenance report
    print(f"\n📊 Maintenance Report:")
    print(f"  Workspaces checked: {maintenance_report['workspaces_checked']}")
    print(f"  Workspaces cleaned: {maintenance_report['workspaces_cleaned']}")
    print(f"  Old workspaces: {len(maintenance_report['old_workspaces'])}")
    print(f"  Oversized workspaces: {len(maintenance_report['oversized_workspaces'])}")
    
    if maintenance_report["errors"]:
        print(f"  Errors: {len(maintenance_report['errors'])}")
        for error in maintenance_report["errors"]:
            print(f"    - {error}")
    
    return maintenance_report

# Run workspace maintenance
maintenance_report = maintain_workspaces("/workspaces", max_age_days=60)
```

### Configuration Management

```python
from cursus.workspace.utils import ConfigUtils, WorkspaceConfig

def manage_workspace_configurations():
    """Demonstrate configuration management capabilities."""
    
    # Base configuration
    base_config = {
        "workspace_settings": {
            "isolation_mode": "strict",
            "auto_cleanup": True,
            "cleanup_threshold_days": 30
        },
        "file_settings": {
            "allowed_extensions": [".py", ".yaml", ".json"],
            "max_file_size_mb": 10
        },
        "logging": {
            "level": "INFO",
            "enable_file_logging": False
        }
    }
    
    # Environment-specific overrides
    development_overrides = {
        "workspace_settings": {
            "cleanup_threshold_days": 7  # Shorter cleanup for dev
        },
        "logging": {
            "level": "DEBUG",
            "enable_file_logging": True
        }
    }
    
    production_overrides = {
        "workspace_settings": {
            "cleanup_threshold_days": 90,  # Longer retention for prod
            "max_workspace_size_mb": 5000
        },
        "file_settings": {
            "max_file_size_mb": 100
        }
    }
    
    # Merge configurations
    dev_config = ConfigUtils.merge_configs(base_config, development_overrides)
    prod_config = ConfigUtils.merge_configs(base_config, production_overrides)
    
    print("Development Configuration:")
    print(f"  Cleanup threshold: {dev_config['workspace_settings']['cleanup_threshold_days']} days")
    print(f"  Logging level: {dev_config['logging']['level']}")
    print(f"  File logging: {dev_config['logging']['enable_file_logging']}")
    
    print("\nProduction Configuration:")
    print(f"  Cleanup threshold: {prod_config['workspace_settings']['cleanup_threshold_days']} days")
    print(f"  Max workspace size: {prod_config['workspace_settings'].get('max_workspace_size_mb', 'unlimited')} MB")
    print(f"  Max file size: {prod_config['file_settings']['max_file_size_mb']} MB")
    
    # Save configurations
    ConfigUtils.save_config(dev_config, "/configs/development.yaml")
    ConfigUtils.save_config(prod_config, "/configs/production.yaml")
    
    print("\n✓ Configurations saved successfully")

# Manage configurations
manage_workspace_configurations()
```

## Integration Points

### Workspace API Integration
The utilities integrate with the high-level WorkspaceAPI to provide underlying functionality for workspace operations.

### Core Management Integration
Utilities are used by the core workspace managers for path operations, configuration management, and validation.

### Template System Integration
File and path utilities support the template system for workspace creation and structure management.

## Related Documentation

- [Workspace API](api.md) - High-level API that uses these utilities
- [Workspace Core](core/README.md) - Core managers that depend on these utilities
- [Workspace Templates](templates.md) - Template system integration
- [Main Workspace Documentation](README.md) - Overview of workspace system

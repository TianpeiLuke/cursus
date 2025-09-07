---
tags:
  - code
  - workspace
  - lifecycle
  - templates
  - management
keywords:
  - WorkspaceLifecycleManager
  - WorkspaceTemplate
  - workspace creation
  - workspace archiving
  - lifecycle management
topics:
  - workspace management
  - lifecycle operations
  - template management
language: python
date of note: 2024-12-07
---

# Workspace Lifecycle Manager

Workspace creation, setup, teardown, and lifecycle operations with comprehensive template support and archiving capabilities.

## Overview

The Workspace Lifecycle Manager provides comprehensive workspace lifecycle management including workspace creation from templates, configuration, archiving, restoration, and cleanup. This module handles the complete lifecycle of workspaces from creation to deletion with proper data preservation and template-based initialization.

The lifecycle system supports workspace creation with template support, structure initialization and validation, configuration and environment setup, archiving and restoration capabilities, and inactive workspace cleanup and maintenance. It provides flexible template management for different workspace types and use cases.

Key features include workspace creation with template support, workspace structure initialization, configuration and environment setup, workspace archiving and restoration, and inactive workspace cleanup and maintenance.

## Classes and Methods

### Classes
- [`WorkspaceTemplate`](#workspacetemplate) - Workspace template for creating new workspaces
- [`WorkspaceLifecycleManager`](#workspacelifecyclemanager) - Workspace lifecycle management with templates and archiving

### Methods
- [`create_workspace`](#create_workspace) - Create a new workspace with optional template
- [`configure_workspace`](#configure_workspace) - Configure an existing workspace
- [`delete_workspace`](#delete_workspace) - Delete a workspace with archiving
- [`archive_workspace`](#archive_workspace) - Archive a workspace without deletion
- [`restore_workspace`](#restore_workspace) - Restore a workspace from archive
- [`cleanup_inactive_workspaces`](#cleanup_inactive_workspaces) - Clean up inactive workspaces
- [`get_available_templates`](#get_available_templates) - Get list of available templates
- [`get_statistics`](#get_statistics) - Get lifecycle management statistics

## API Reference

### WorkspaceTemplate

_class_ cursus.workspace.core.lifecycle.WorkspaceTemplate(_template_name_, _template_path_)

Workspace template for creating new workspaces with predefined structure and configuration.

**Parameters:**
- **template_name** (_str_) – Name of the template.
- **template_path** (_Path_) – Path to template directory containing structure and metadata.

```python
from cursus.workspace.core.lifecycle import WorkspaceTemplate
from pathlib import Path

# Create workspace template
template = WorkspaceTemplate(
    template_name="ml_workspace",
    template_path=Path("/templates/ml_workspace")
)

print("Template name:", template.template_name)
print("Template metadata:", template.metadata)
```

### WorkspaceLifecycleManager

_class_ cursus.workspace.core.lifecycle.WorkspaceLifecycleManager(_workspace_manager_)

Workspace lifecycle management with comprehensive template support and archiving capabilities.

**Parameters:**
- **workspace_manager** (_WorkspaceManager_) – Parent WorkspaceManager instance for integration.

```python
from cursus.workspace.core.lifecycle import WorkspaceLifecycleManager
from cursus.workspace.core.manager import WorkspaceManager

# Create lifecycle manager
workspace_manager = WorkspaceManager("/path/to/workspace")
lifecycle_manager = WorkspaceLifecycleManager(workspace_manager)

# Get available templates
templates = lifecycle_manager.get_available_templates()
print("Available templates:", [t['name'] for t in templates])
```

#### create_workspace

create_workspace(_developer_id_, _workspace_type="developer"_, _template=None_, _create_structure=True_, _**kwargs_)

Create a new workspace with optional template and configuration.

**Parameters:**
- **developer_id** (_str_) – Developer identifier for the workspace.
- **workspace_type** (_str_) – Type of workspace ('developer', 'shared', 'test'), defaults to 'developer'.
- **template** (_Optional[str]_) – Optional template name to use for workspace creation.
- **create_structure** (_bool_) – Whether to create directory structure, defaults to True.
- ****kwargs** – Additional workspace configuration parameters.

**Returns:**
- **WorkspaceContext** – WorkspaceContext for the created workspace.

```python
# Create basic developer workspace
workspace_context = lifecycle_manager.create_workspace(
    developer_id="alice",
    workspace_type="developer"
)

print("Created workspace:", workspace_context.workspace_id)
print("Workspace path:", workspace_context.workspace_path)

# Create workspace with template
ml_workspace = lifecycle_manager.create_workspace(
    developer_id="bob",
    workspace_type="developer",
    template="ml_workspace",
    additional_config={"enable_gpu": True}
)

# Create shared workspace
shared_workspace = lifecycle_manager.create_workspace(
    developer_id="shared",
    workspace_type="shared",
    template="basic"
)

# Create test workspace
test_workspace = lifecycle_manager.create_workspace(
    developer_id="alice",
    workspace_type="test",
    test_environment="staging"
)
```

#### configure_workspace

configure_workspace(_workspace_id_, _config_)

Configure an existing workspace with new settings and metadata.

**Parameters:**
- **workspace_id** (_str_) – Workspace identifier to configure.
- **config** (_Dict[str, Any]_) – Configuration dictionary with workspace settings.

**Returns:**
- **WorkspaceContext** – Updated WorkspaceContext with new configuration.

```python
# Configure workspace with additional settings
config = {
    "enable_additional_modules": True,
    "additional_modules": ["validators", "transformers"],
    "gpu_enabled": True,
    "max_memory": "16GB",
    "environment_variables": {
        "CUDA_VISIBLE_DEVICES": "0,1"
    }
}

updated_context = lifecycle_manager.configure_workspace("alice", config)
print("Updated workspace:", updated_context.workspace_id)
print("Configuration applied:", updated_context.metadata)
```

#### delete_workspace

delete_workspace(_workspace_id_)

Delete a workspace with automatic archiving of user data.

**Parameters:**
- **workspace_id** (_str_) – Workspace identifier to delete.

**Returns:**
- **bool** – True if deletion was successful.

```python
# Delete workspace (automatically archives if contains data)
try:
    success = lifecycle_manager.delete_workspace("alice")
    if success:
        print("Workspace deleted successfully")
        print("User data was automatically archived")
    else:
        print("Failed to delete workspace")
except ValueError as e:
    print(f"Deletion error: {e}")
```

#### archive_workspace

archive_workspace(_workspace_id_)

Archive a workspace without deleting it for backup purposes.

**Parameters:**
- **workspace_id** (_str_) – Workspace identifier to archive.

**Returns:**
- **Dict[str, Any]** – Archive result with success status, archive path, and timestamp.

```python
# Archive workspace for backup
archive_result = lifecycle_manager.archive_workspace("alice")

if archive_result['success']:
    print("Workspace archived successfully")
    print("Archive path:", archive_result['archive_path'])
    print("Archived at:", archive_result['archived_at'])
else:
    print("Archive failed:", archive_result['error'])
```

#### restore_workspace

restore_workspace(_workspace_id_, _archive_path_)

Restore a workspace from archive with full structure recreation.

**Parameters:**
- **workspace_id** (_str_) – Workspace identifier for restoration.
- **archive_path** (_str_) – Path to archived workspace directory.

**Returns:**
- **Dict[str, Any]** – Restore result with success status, restore path, and timestamp.

```python
# Restore workspace from archive
restore_result = lifecycle_manager.restore_workspace(
    workspace_id="alice",
    archive_path="/workspace/archived_workspaces/alice_20241207_120000"
)

if restore_result['success']:
    print("Workspace restored successfully")
    print("Restore path:", restore_result['restore_path'])
    print("Restored at:", restore_result['restored_at'])
else:
    print("Restore failed:", restore_result['error'])
```

#### cleanup_inactive_workspaces

cleanup_inactive_workspaces(_inactive_threshold=timedelta(days=30)_)

Clean up inactive workspaces with configurable inactivity threshold.

**Parameters:**
- **inactive_threshold** (_timedelta_) – Threshold for considering workspace inactive, defaults to 30 days.

**Returns:**
- **Dict[str, Any]** – Cleanup result with lists of cleaned up, archived, and error workspaces.

```python
from datetime import timedelta

# Clean up workspaces inactive for more than 30 days
cleanup_result = lifecycle_manager.cleanup_inactive_workspaces(
    inactive_threshold=timedelta(days=30)
)

print("Cleanup summary:")
print(f"  Cleaned up: {len(cleanup_result['cleaned_up'])} workspaces")
print(f"  Archived: {len(cleanup_result['archived'])} workspaces")
print(f"  Errors: {len(cleanup_result['errors'])} workspaces")
print(f"  Total processed: {cleanup_result['total_processed']}")

# Clean up with custom threshold (7 days)
weekly_cleanup = lifecycle_manager.cleanup_inactive_workspaces(
    inactive_threshold=timedelta(days=7)
)

# Review archived workspaces
for archived in cleanup_result['archived']:
    print(f"Archived {archived['workspace_id']} to {archived['archive_path']}")
```

#### get_available_templates

get_available_templates()

Get list of available workspace templates with metadata.

**Returns:**
- **List[Dict[str, Any]]** – List of template information dictionaries with names, paths, and metadata.

```python
# Get all available templates
templates = lifecycle_manager.get_available_templates()

print("Available workspace templates:")
for template in templates:
    print(f"  Name: {template['name']}")
    print(f"  Path: {template['path']}")
    print(f"  Description: {template['metadata'].get('description', 'No description')}")
    print(f"  Version: {template['metadata'].get('version', 'Unknown')}")
    print("---")

# Filter templates by type
ml_templates = [t for t in templates if 'ml' in t['name'].lower()]
print(f"ML templates available: {len(ml_templates)}")
```

#### get_statistics

get_statistics()

Get comprehensive lifecycle management statistics.

**Returns:**
- **Dict[str, Any]** – Statistics including template counts, workspace operations, and status distribution.

```python
# Get lifecycle management statistics
stats = lifecycle_manager.get_statistics()

print("Lifecycle Statistics:")
print(f"Available templates: {stats['available_templates']}")
print(f"Template names: {stats['template_names']}")

workspace_ops = stats['workspace_operations']
print(f"Total workspaces: {workspace_ops['total_workspaces']}")
print(f"Active workspaces: {workspace_ops['active_workspaces']}")
print(f"Archived workspaces: {workspace_ops['archived_workspaces']}")
```

## Lifecycle Workflow

### Complete Workspace Creation

```python
# Complete workspace creation workflow
def create_workspace_with_setup(lifecycle_manager, developer_id, workspace_type="developer"):
    try:
        # 1. Create workspace with template
        workspace_context = lifecycle_manager.create_workspace(
            developer_id=developer_id,
            workspace_type=workspace_type,
            template="ml_workspace"
        )
        
        print(f"✅ Created workspace: {workspace_context.workspace_id}")
        
        # 2. Configure workspace
        config = {
            "enable_additional_modules": True,
            "additional_modules": ["validators", "transformers"],
            "environment": "development",
            "created_by": "lifecycle_manager"
        }
        
        configured_context = lifecycle_manager.configure_workspace(
            workspace_context.workspace_id, config
        )
        
        print(f"✅ Configured workspace with {len(config)} settings")
        
        # 3. Verify workspace structure
        workspace_path = Path(configured_context.workspace_path)
        if workspace_path.exists():
            print(f"✅ Workspace structure created at: {workspace_path}")
            
            # List created directories
            cursus_dev_dir = workspace_path / "src" / "cursus_dev" / "steps"
            if cursus_dev_dir.exists():
                module_dirs = [d.name for d in cursus_dev_dir.iterdir() if d.is_dir()]
                print(f"✅ Module directories: {module_dirs}")
        
        return configured_context
        
    except Exception as e:
        print(f"❌ Workspace creation failed: {e}")
        return None
```

### Workspace Backup and Restore

```python
# Backup and restore workflow
def backup_and_restore_workspace(lifecycle_manager, workspace_id):
    try:
        # 1. Archive workspace
        print(f"Backing up workspace: {workspace_id}")
        archive_result = lifecycle_manager.archive_workspace(workspace_id)
        
        if not archive_result['success']:
            print(f"❌ Backup failed: {archive_result['error']}")
            return False
        
        archive_path = archive_result['archive_path']
        print(f"✅ Workspace backed up to: {archive_path}")
        
        # 2. Simulate workspace deletion (for testing restore)
        print(f"Simulating workspace deletion...")
        # In real scenario, you might delete the workspace here
        
        # 3. Restore workspace
        print(f"Restoring workspace from backup...")
        restore_result = lifecycle_manager.restore_workspace(workspace_id, archive_path)
        
        if restore_result['success']:
            print(f"✅ Workspace restored to: {restore_result['restore_path']}")
            return True
        else:
            print(f"❌ Restore failed: {restore_result['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Backup/restore workflow failed: {e}")
        return False
```

### Workspace Maintenance

```python
# Regular workspace maintenance
def perform_workspace_maintenance(lifecycle_manager):
    print("🔧 Starting workspace maintenance...")
    
    # 1. Get current statistics
    stats = lifecycle_manager.get_statistics()
    print(f"Current state: {stats['workspace_operations']['total_workspaces']} total workspaces")
    
    # 2. Clean up inactive workspaces (older than 30 days)
    cleanup_result = lifecycle_manager.cleanup_inactive_workspaces(
        inactive_threshold=timedelta(days=30)
    )
    
    print(f"Maintenance results:")
    print(f"  🗑️  Cleaned up: {len(cleanup_result['cleaned_up'])} workspaces")
    print(f"  📦 Archived: {len(cleanup_result['archived'])} workspaces")
    print(f"  ❌ Errors: {len(cleanup_result['errors'])} workspaces")
    
    # 3. Report any errors
    if cleanup_result['errors']:
        print("Maintenance errors:")
        for error in cleanup_result['errors']:
            print(f"  - {error}")
    
    # 4. Get updated statistics
    updated_stats = lifecycle_manager.get_statistics()
    print(f"After maintenance: {updated_stats['workspace_operations']['total_workspaces']} total workspaces")
    
    return cleanup_result
```

### Template Management

```python
# Template management workflow
def manage_workspace_templates(lifecycle_manager):
    # 1. List available templates
    templates = lifecycle_manager.get_available_templates()
    print(f"Available templates: {len(templates)}")
    
    for template in templates:
        print(f"  📋 {template['name']}")
        print(f"     Description: {template['metadata'].get('description', 'No description')}")
        print(f"     Version: {template['metadata'].get('version', 'Unknown')}")
    
    # 2. Create workspaces with different templates
    template_usage = {}
    
    for template in templates:
        template_name = template['name']
        try:
            # Create test workspace with template
            test_workspace = lifecycle_manager.create_workspace(
                developer_id=f"test_{template_name}",
                workspace_type="test",
                template=template_name
            )
            
            template_usage[template_name] = "success"
            print(f"✅ Successfully created workspace with {template_name} template")
            
        except Exception as e:
            template_usage[template_name] = f"error: {e}"
            print(f"❌ Failed to create workspace with {template_name} template: {e}")
    
    return template_usage
```

### Error Handling and Recovery

```python
# Robust workspace operations with error handling
def safe_workspace_operations(lifecycle_manager, developer_id):
    operations_log = []
    
    try:
        # 1. Safe workspace creation
        print(f"Creating workspace for {developer_id}...")
        workspace_context = lifecycle_manager.create_workspace(
            developer_id=developer_id,
            workspace_type="developer"
        )
        operations_log.append(f"✅ Created workspace: {workspace_context.workspace_id}")
        
        # 2. Safe configuration
        try:
            config = {"environment": "development", "auto_cleanup": True}
            lifecycle_manager.configure_workspace(workspace_context.workspace_id, config)
            operations_log.append("✅ Applied workspace configuration")
        except Exception as config_error:
            operations_log.append(f"⚠️  Configuration failed: {config_error}")
        
        # 3. Safe archiving
        try:
            archive_result = lifecycle_manager.archive_workspace(workspace_context.workspace_id)
            if archive_result['success']:
                operations_log.append(f"✅ Archived workspace: {archive_result['archive_path']}")
            else:
                operations_log.append(f"⚠️  Archive failed: {archive_result['error']}")
        except Exception as archive_error:
            operations_log.append(f"⚠️  Archive error: {archive_error}")
        
        return True, operations_log
        
    except Exception as e:
        operations_log.append(f"❌ Critical error: {e}")
        
        # Attempt cleanup if workspace was partially created
        try:
            if 'workspace_context' in locals():
                lifecycle_manager.delete_workspace(workspace_context.workspace_id)
                operations_log.append("🧹 Cleaned up partially created workspace")
        except Exception as cleanup_error:
            operations_log.append(f"❌ Cleanup failed: {cleanup_error}")
        
        return False, operations_log
```

## Related Documentation

- [Workspace Manager](manager.md) - Consolidated workspace management system
- [Workspace Templates](../templates.md) - Template system and built-in templates
- [Workspace Configuration](config.md) - Pipeline and step configuration models
- [Workspace API](../api.md) - High-level workspace API interface
- [Workspace Utils](../utils.md) - Workspace utility functions and classes

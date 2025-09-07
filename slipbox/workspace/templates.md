---
tags:
  - code
  - workspace
  - templates
  - scaffolding
  - automation
keywords:
  - WorkspaceTemplate
  - TemplateManager
  - TemplateType
  - workspace scaffolding
  - template system
  - workspace creation
topics:
  - workspace templates
  - workspace scaffolding
  - template management
  - workspace automation
language: python
date of note: 2024-12-07
---

# Workspace Templates

Template system for standardized workspace creation, providing pre-configured workspace structures for different development scenarios and use cases.

## Overview

The Workspace Templates module provides a comprehensive template management system for creating standardized workspace structures. It supports multiple template types including basic development workspaces, machine learning pipelines, and data processing environments, with extensible custom template creation capabilities.

The system includes built-in templates for common development scenarios, template management utilities for creating and applying custom templates, and integration with the workspace creation workflow. Templates define directory structures, initial files, dependencies, and configuration overrides to provide consistent, ready-to-use development environments.

## Classes and Methods

### Classes
- [`TemplateType`](#templatetype) - Enumeration of available workspace template types
- [`WorkspaceTemplate`](#workspacetemplate) - Workspace template configuration with structure and metadata
- [`TemplateManager`](#templatemanager) - Template management system for creation and application

## API Reference

### TemplateType

_class_ cursus.workspace.templates.TemplateType(_Enum_)

Enumeration defining available workspace template types for different development scenarios.

**Values:**
- **BASIC** – Basic workspace with standard directory structure
- **ML_PIPELINE** – Machine learning pipeline workspace with ML-specific structure
- **DATA_PROCESSING** – Data processing workspace for ETL and transformation pipelines
- **CUSTOM** – Custom user-defined template

```python
from cursus.workspace.templates import TemplateType

# Use template types
basic_type = TemplateType.BASIC
ml_type = TemplateType.ML_PIPELINE
data_type = TemplateType.DATA_PROCESSING
custom_type = TemplateType.CUSTOM
```

### WorkspaceTemplate

_class_ cursus.workspace.templates.WorkspaceTemplate(_name_, _type_, _description=""_, _directories=[]_, _files={}_, _dependencies=[]_, _config_overrides={}_, _created_at=None_, _version="1.0.0"_)

Workspace template configuration defining structure, files, dependencies, and metadata for workspace creation.

**Parameters:**
- **name** (_str_) – Template name (minimum length 1)
- **type** (_TemplateType_) – Type of template
- **description** (_str_) – Template description (default empty)
- **directories** (_List[str]_) – List of directories to create
- **files** (_Dict[str, str]_) – Dictionary mapping file paths to content
- **dependencies** (_List[str]_) – List of required dependencies
- **config_overrides** (_Dict[str, Any]_) – Default configuration overrides
- **created_at** (_Optional[str]_) – Template creation timestamp
- **version** (_str_) – Template version (default "1.0.0")

```python
from cursus.workspace.templates import WorkspaceTemplate, TemplateType

# Create basic template
template = WorkspaceTemplate(
    name="my_basic_template",
    type=TemplateType.BASIC,
    description="Custom basic workspace template",
    directories=["src", "tests", "docs"],
    files={
        "README.md": "# My Workspace\n\nCustom workspace template",
        "src/__init__.py": "",
        "tests/__init__.py": ""
    },
    dependencies=["pytest", "black", "flake8"]
)
```

### TemplateManager

_class_ cursus.workspace.templates.TemplateManager(_templates_dir=None_)

Template management system for creating, storing, retrieving, and applying workspace templates.

**Parameters:**
- **templates_dir** (_Optional[Path]_) – Directory containing template definitions (defaults to built-in templates directory)

```python
from cursus.workspace.templates import TemplateManager

# Initialize with default template directory
manager = TemplateManager()

# Initialize with custom template directory
manager = TemplateManager(Path("/custom/templates"))
```

#### get_template

get_template(_name_)

Retrieve a template by name from the template directory.

**Parameters:**
- **name** (_str_) – Template name to retrieve

**Returns:**
- **Optional[WorkspaceTemplate]** – Template if found, None otherwise

```python
# Get built-in template
template = manager.get_template("ml_pipeline")
if template:
    print(f"Template: {template.description}")
    print(f"Directories: {template.directories}")
else:
    print("Template not found")
```

#### list_templates

list_templates()

List all available templates in the template directory.

**Returns:**
- **List[WorkspaceTemplate]** – List of all available templates

```python
templates = manager.list_templates()
print(f"Available templates ({len(templates)}):")

for template in templates:
    print(f"  {template.name} ({template.type.value}): {template.description}")
```

#### create_template

create_template(_template_)

Create a new template and save it to the template directory.

**Parameters:**
- **template** (_WorkspaceTemplate_) – Template to create and save

**Returns:**
- **bool** – True if template was created successfully, False otherwise

```python
# Create custom template
custom_template = WorkspaceTemplate(
    name="data_science_advanced",
    type=TemplateType.ML_PIPELINE,
    description="Advanced data science workspace with GPU support",
    directories=[
        "data/raw", "data/processed", "data/features",
        "models", "notebooks", "reports", "scripts"
    ],
    files={
        "README.md": "# Advanced Data Science Workspace",
        "requirements.txt": "pandas\nnumpy\nscikit-learn\ntensorflow-gpu",
        "notebooks/exploration.ipynb": "# Exploration notebook content"
    },
    dependencies=["pandas", "numpy", "scikit-learn", "tensorflow-gpu"],
    config_overrides={"enable_gpu": True, "memory_limit": "16GB"}
)

# Save template
success = manager.create_template(custom_template)
if success:
    print("Template created successfully")
else:
    print("Failed to create template")
```

#### apply_template

apply_template(_template_name_, _workspace_path_)

Apply a template to a workspace directory, creating the defined structure and files.

**Parameters:**
- **template_name** (_str_) – Name of template to apply
- **workspace_path** (_Path_) – Path to workspace directory where template should be applied

**Returns:**
- **bool** – True if template was applied successfully, False otherwise

```python
from pathlib import Path

# Apply template to workspace
workspace_path = Path("./new_workspace")
success = manager.apply_template("ml_pipeline", workspace_path)

if success:
    print(f"Template applied to {workspace_path}")
    
    # Verify structure was created
    if (workspace_path / "data").exists():
        print("✓ Data directory created")
    if (workspace_path / "models").exists():
        print("✓ Models directory created")
    if (workspace_path / "notebooks").exists():
        print("✓ Notebooks directory created")
else:
    print("Failed to apply template")
```

## Built-in Templates

### Basic Template

The basic template provides a standard workspace structure suitable for general development:

```python
# Basic template structure
directories = [
    "builders",      # Step builder implementations
    "configs",       # Configuration classes
    "contracts",     # Step contracts
    "specs",         # Step specifications
    "scripts",       # Pipeline scripts
    "tests"          # Test files
]

files = {
    "README.md": "# Developer Workspace\n\nBasic workspace for Cursus pipeline development",
    ".gitignore": "# Python gitignore content",
    "builders/__init__.py": "",
    "configs/__init__.py": "",
    "contracts/__init__.py": "",
    "specs/__init__.py": "",
    "scripts/__init__.py": "",
    "tests/__init__.py": "",
    "tests/test_example.py": "# Example test file"
}
```

### ML Pipeline Template

The ML pipeline template provides structure optimized for machine learning development:

```python
# ML pipeline template structure
directories = [
    "builders", "configs", "contracts", "specs", "scripts",
    "data/raw", "data/processed", "data/features",
    "models", "notebooks", "tests"
]

files = {
    "README.md": "# ML Pipeline Workspace",
    ".gitignore": "# ML-specific gitignore with data and model exclusions",
    "notebooks/exploration.ipynb": "# Jupyter notebook for data exploration",
    "data/README.md": "# Data directory documentation",
    "models/README.md": "# Model artifacts documentation"
}

dependencies = ["pandas", "numpy", "scikit-learn", "jupyter"]
```

### Data Processing Template

The data processing template provides structure for ETL and data transformation pipelines:

```python
# Data processing template structure
directories = [
    "builders", "configs", "contracts", "specs", "scripts",
    "data/raw", "data/processed", "data/output",
    "schemas", "tests"
]

files = {
    "README.md": "# Data Processing Workspace",
    "schemas/README.md": "# Data schemas and validation rules",
    "data/README.md": "# Data flow documentation"
}

dependencies = ["pandas", "pydantic", "jsonschema"]
```

## Usage Examples

### Template Discovery and Selection

```python
from cursus.workspace.templates import TemplateManager

# Initialize template manager
manager = TemplateManager()

# List available templates
templates = manager.list_templates()
print("Available templates:")

for template in templates:
    print(f"\n{template.name} ({template.type.value})")
    print(f"  Description: {template.description}")
    print(f"  Directories: {len(template.directories)}")
    print(f"  Files: {len(template.files)}")
    print(f"  Dependencies: {len(template.dependencies)}")

# Get specific template details
ml_template = manager.get_template("ml_pipeline")
if ml_template:
    print(f"\nML Pipeline Template Details:")
    print(f"  Version: {ml_template.version}")
    print(f"  Created: {ml_template.created_at}")
    print(f"  Directories: {', '.join(ml_template.directories)}")
```

### Custom Template Creation

```python
from cursus.workspace.templates import WorkspaceTemplate, TemplateType, TemplateManager

# Create specialized template for computer vision projects
cv_template = WorkspaceTemplate(
    name="computer_vision",
    type=TemplateType.ML_PIPELINE,
    description="Computer vision workspace with image processing tools",
    directories=[
        "data/images/raw",
        "data/images/processed", 
        "data/annotations",
        "models/checkpoints",
        "models/exports",
        "notebooks",
        "scripts/preprocessing",
        "scripts/training",
        "scripts/inference",
        "tests"
    ],
    files={
        "README.md": """# Computer Vision Workspace

Specialized workspace for computer vision projects.

## Directory Structure
- `data/images/` - Image datasets
- `data/annotations/` - Image annotations and labels
- `models/` - Trained models and checkpoints
- `scripts/` - Processing and training scripts
""",
        "requirements.txt": """opencv-python
pillow
torch
torchvision
albumentations
matplotlib
seaborn""",
        "scripts/preprocessing/image_utils.py": """# Image preprocessing utilities
import cv2
import numpy as np

def resize_image(image, target_size):
    return cv2.resize(image, target_size)
""",
        "notebooks/data_exploration.ipynb": """# Computer vision data exploration notebook""",
        ".gitignore": """# CV-specific gitignore
data/images/raw/*
!data/images/raw/.gitkeep
models/checkpoints/*
!models/checkpoints/.gitkeep
*.pth
*.onnx
"""
    },
    dependencies=[
        "opencv-python", "pillow", "torch", "torchvision", 
        "albumentations", "matplotlib", "seaborn"
    ],
    config_overrides={
        "enable_gpu": True,
        "image_formats": [".jpg", ".png", ".tiff"],
        "batch_size": 32
    }
)

# Save the custom template
manager = TemplateManager()
success = manager.create_template(cv_template)

if success:
    print("Computer vision template created successfully")
    
    # Apply to new workspace
    from pathlib import Path
    workspace_path = Path("./cv_project")
    if manager.apply_template("computer_vision", workspace_path):
        print(f"Template applied to {workspace_path}")
```

### Template Application Workflow

```python
from pathlib import Path
from cursus.workspace.templates import TemplateManager

def setup_workspace_from_template(workspace_name, template_name, base_path="./workspaces"):
    """Complete workflow for setting up workspace from template."""
    
    manager = TemplateManager()
    workspace_path = Path(base_path) / workspace_name
    
    # Check if template exists
    template = manager.get_template(template_name)
    if not template:
        print(f"Template '{template_name}' not found")
        return False
    
    # Create workspace directory
    workspace_path.mkdir(parents=True, exist_ok=True)
    
    # Apply template
    success = manager.apply_template(template_name, workspace_path)
    
    if success:
        print(f"✓ Workspace '{workspace_name}' created from template '{template_name}'")
        print(f"  Location: {workspace_path.absolute()}")
        print(f"  Template type: {template.type.value}")
        print(f"  Dependencies: {', '.join(template.dependencies)}")
        
        # Show created structure
        print("\nCreated structure:")
        for directory in template.directories:
            dir_path = workspace_path / directory
            if dir_path.exists():
                print(f"  📁 {directory}/")
        
        for file_path in template.files.keys():
            file_full_path = workspace_path / file_path
            if file_full_path.exists():
                print(f"  📄 {file_path}")
        
        return True
    else:
        print(f"✗ Failed to create workspace from template")
        return False

# Use the workflow
setup_workspace_from_template("ml_project_1", "ml_pipeline")
setup_workspace_from_template("data_pipeline_1", "data_processing")
setup_workspace_from_template("cv_project_1", "computer_vision")
```

### Template Validation and Testing

```python
from cursus.workspace.templates import TemplateManager, WorkspaceTemplate
from pathlib import Path
import tempfile
import shutil

def validate_template(template_name):
    """Validate a template by applying it to a temporary directory."""
    
    manager = TemplateManager()
    template = manager.get_template(template_name)
    
    if not template:
        print(f"Template '{template_name}' not found")
        return False
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "test_workspace"
        
        # Apply template
        success = manager.apply_template(template_name, temp_path)
        
        if not success:
            print(f"✗ Failed to apply template '{template_name}'")
            return False
        
        # Validate structure
        validation_results = {
            "directories_created": 0,
            "files_created": 0,
            "missing_directories": [],
            "missing_files": []
        }
        
        # Check directories
        for directory in template.directories:
            dir_path = temp_path / directory
            if dir_path.exists() and dir_path.is_dir():
                validation_results["directories_created"] += 1
            else:
                validation_results["missing_directories"].append(directory)
        
        # Check files
        for file_path in template.files.keys():
            full_file_path = temp_path / file_path
            if full_file_path.exists() and full_file_path.is_file():
                validation_results["files_created"] += 1
            else:
                validation_results["missing_files"].append(file_path)
        
        # Report results
        print(f"Template '{template_name}' validation:")
        print(f"  ✓ Directories created: {validation_results['directories_created']}/{len(template.directories)}")
        print(f"  ✓ Files created: {validation_results['files_created']}/{len(template.files)}")
        
        if validation_results["missing_directories"]:
            print(f"  ✗ Missing directories: {validation_results['missing_directories']}")
        
        if validation_results["missing_files"]:
            print(f"  ✗ Missing files: {validation_results['missing_files']}")
        
        is_valid = (
            len(validation_results["missing_directories"]) == 0 and
            len(validation_results["missing_files"]) == 0
        )
        
        print(f"  Overall: {'✓ VALID' if is_valid else '✗ INVALID'}")
        return is_valid

# Validate all built-in templates
manager = TemplateManager()
templates = manager.list_templates()

print("Validating all templates:")
for template in templates:
    validate_template(template.name)
    print()
```

### Template Customization and Extension

```python
from cursus.workspace.templates import TemplateManager, WorkspaceTemplate, TemplateType

def extend_template(base_template_name, new_template_name, additional_config):
    """Extend an existing template with additional configuration."""
    
    manager = TemplateManager()
    base_template = manager.get_template(base_template_name)
    
    if not base_template:
        print(f"Base template '{base_template_name}' not found")
        return False
    
    # Create extended template
    extended_template = WorkspaceTemplate(
        name=new_template_name,
        type=base_template.type,
        description=f"Extended {base_template.description}",
        directories=base_template.directories + additional_config.get("directories", []),
        files={**base_template.files, **additional_config.get("files", {})},
        dependencies=base_template.dependencies + additional_config.get("dependencies", []),
        config_overrides={**base_template.config_overrides, **additional_config.get("config_overrides", {})},
        version="1.0.0"
    )
    
    # Save extended template
    success = manager.create_template(extended_template)
    
    if success:
        print(f"✓ Extended template '{new_template_name}' created from '{base_template_name}'")
        return True
    else:
        print(f"✗ Failed to create extended template")
        return False

# Extend ML pipeline template for deep learning
deep_learning_config = {
    "directories": [
        "data/tfrecords",
        "models/tensorboard",
        "configs/experiments"
    ],
    "files": {
        "configs/model_config.yaml": """# Model configuration
model:
  type: "neural_network"
  layers: 3
  activation: "relu"
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
""",
        "scripts/train_model.py": """# Training script template
import tensorflow as tf

def train_model():
    # Training implementation
    pass
"""
    },
    "dependencies": ["tensorflow", "tensorboard", "keras"],
    "config_overrides": {
        "enable_gpu": True,
        "mixed_precision": True,
        "distributed_training": False
    }
}

extend_template("ml_pipeline", "deep_learning", deep_learning_config)
```

## Integration Points

### Workspace API Integration
Templates integrate seamlessly with the WorkspaceAPI for automated workspace creation with pre-configured structures.

### CLI Integration
Template management is accessible through the Cursus CLI for command-line template operations and workspace creation.

### Configuration System Integration
Templates support configuration overrides that integrate with the workspace configuration management system.

## Related Documentation

- [Workspace API](api.md) - High-level workspace API that uses templates
- [Workspace Core](core/README.md) - Core workspace management functionality
- [Main Workspace Documentation](README.md) - Overview of workspace management system
- [CLI Integration](../cli/workspace_cli.md) - Command-line interface for template operations

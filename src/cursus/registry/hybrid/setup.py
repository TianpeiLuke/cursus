"""
Workspace registry initialization utilities.
Simplified implementation following redundancy evaluation guide principles.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import os


def create_workspace_registry(
    workspace_path: str, developer_id: str, template: str = "standard"
) -> str:
    """Create simple workspace registry structure for a developer.

    Args:
        workspace_path: Path to the developer workspace
        developer_id: Unique identifier for the developer
        template: Registry template type (standard/minimal)

    Returns:
        Path to the created registry file

    Raises:
        ValueError: If developer_id is invalid or workspace creation fails
    """
    # Validate developer ID
    if not developer_id or not developer_id.strip():
        raise ValueError("Developer ID cannot be empty")

    if not developer_id.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Developer ID '{developer_id}' contains invalid characters")

    workspace_dir = Path(workspace_path)
    registry_dir = workspace_dir / "src" / "cursus_dev" / "registry"

    # Create registry directory
    registry_dir.mkdir(parents=True, exist_ok=True)

    # Create __init__.py
    init_file = registry_dir / "__init__.py"
    init_file.write_text('"""Local registry for workspace."""\n')

    # Create workspace_registry.py from template
    registry_file = registry_dir / "workspace_registry.py"
    template_content = _get_registry_template(developer_id, template)
    registry_file.write_text(template_content)

    return str(registry_file)


def _get_registry_template(developer_id: str, template: str) -> str:
    """Get simplified registry template content.

    Following redundancy evaluation guide - removed complex metadata
    that addresses theoretical conflicts without validated demand.
    """
    if template == "minimal":
        return _get_minimal_template(developer_id)
    else:
        return _get_standard_template(developer_id)


def _get_standard_template(developer_id: str) -> str:
    """Get standard registry template with essential fields only."""
    return f'''"""
Local registry for {developer_id} workspace.
Simple format following redundancy evaluation guide principles.
"""

# Local step definitions (new steps specific to this workspace)
LOCAL_STEPS = {{
    # Add your custom steps here
    # Example:
    # "MyCustomProcessingStep": {{
    #     "config_class": "MyCustomProcessingConfig",
    #     "builder_step_name": "MyCustomProcessingStepBuilder",
    #     "spec_type": "MyCustomProcessing",
    #     "sagemaker_step_type": "Processing",
    #     "description": "Custom processing step for {developer_id}"
    # }},
    
    # "ExperimentalTrainingStep": {{
    #     "config_class": "ExperimentalTrainingConfig",
    #     "builder_step_name": "ExperimentalTrainingStepBuilder",
    #     "spec_type": "ExperimentalTraining",
    #     "sagemaker_step_type": "Training",
    #     "description": "Experimental training approach"
    # }}
}}

# Step overrides (override core step definitions for this workspace)
STEP_OVERRIDES = {{
    # Override core steps here if needed
    # Example:
    # "XGBoostTraining": {{
    #     "config_class": "CustomXGBoostTrainingConfig",
    #     "builder_step_name": "CustomXGBoostTrainingStepBuilder",
    #     "spec_type": "CustomXGBoostTraining",
    #     "sagemaker_step_type": "Training",
    #     "description": "Custom XGBoost implementation with enhanced features"
    # }}
}}

# Simple workspace metadata
WORKSPACE_METADATA = {{
    "developer_id": "{developer_id}",
    "version": "1.0.0",
    "description": "Custom ML pipeline extensions"
}}
'''


def _get_minimal_template(developer_id: str) -> str:
    """Get minimal registry template with bare essentials."""
    return f'''"""
Minimal local registry for {developer_id} workspace.
"""

# Local step definitions
LOCAL_STEPS = {{
    # Add your custom steps here
}}

# Step overrides
STEP_OVERRIDES = {{
    # Override core steps here if needed
}}

# Workspace metadata
WORKSPACE_METADATA = {{
    "developer_id": "{developer_id}",
    "version": "1.0.0"
}}
'''


def create_workspace_structure(workspace_path: str) -> None:
    """Create complete workspace directory structure."""
    workspace_dir = Path(workspace_path)

    directories = [
        "src/cursus_dev/steps/builders",
        "src/cursus_dev/steps/configs",
        "src/cursus_dev/steps/contracts",
        "src/cursus_dev/steps/scripts",
        "src/cursus_dev/steps/specs",
        "src/cursus_dev/registry",
        "test/unit",
        "test/integration",
        "validation_reports",
        "examples",
        "docs",
    ]

    for dir_path in directories:
        full_path = workspace_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)

        # Create __init__.py files for Python packages
        if "src/cursus_dev" in dir_path:
            init_file = full_path / "__init__.py"
            init_file.write_text('"""Package initialization."""\n')


def create_workspace_documentation(
    workspace_dir: Path, developer_id: str, registry_file: str
) -> Path:
    """Create comprehensive workspace documentation."""
    readme_file = workspace_dir / "README.md"
    readme_content = f"""# Developer Workspace: {developer_id}

This workspace contains custom step implementations for developer {developer_id}.

## Directory Structure

```
{developer_id}/
├── src/cursus_dev/           # Custom step implementations
│   ├── steps/
│   │   ├── builders/         # Step builder classes
│   │   ├── configs/          # Configuration classes
│   │   ├── contracts/        # Script contracts
│   │   ├── scripts/          # Processing scripts
│   │   └── specs/            # Step specifications
│   └── registry/             # Local registry
│       └── workspace_registry.py
├── test/                     # Unit and integration tests
├── validation_reports/       # Validation results
├── examples/                 # Usage examples
└── docs/                     # Additional documentation
```

## Registry

Local registry: `{registry_file}`

## Quick Start

### 1. Set Workspace Context
```bash
export CURSUS_WORKSPACE_ID={developer_id}
```

### 2. Add Custom Steps
Edit `{registry_file}` to define your custom steps:

```python
LOCAL_STEPS = {{
    "MyCustomStep": {{
        "config_class": "MyCustomStepConfig",
        "builder_step_name": "MyCustomStepBuilder",
        "spec_type": "MyCustomStep",
        "sagemaker_step_type": "Processing",
        "description": "My custom processing step"
    }}
}}
```

### 3. Implement Step Components
Create the corresponding implementation files:
- Config: `src/cursus_dev/steps/configs/my_custom_step_config.py`
- Builder: `src/cursus_dev/steps/builders/my_custom_step_builder.py`
- Contract: `src/cursus_dev/steps/contracts/my_custom_step_contract.py`
- Script: `src/cursus_dev/steps/scripts/my_custom_step_script.py`
- Spec: `src/cursus_dev/steps/specs/my_custom_step_spec.py`

### 4. Test Your Implementation
```python
from cursus.registry import set_workspace_context, get_config_class_name

set_workspace_context("{developer_id}")
config_class = get_config_class_name("MyCustomStep")  # Uses your local registry
```

## CLI Commands

### Registry Management
```bash
# List steps in this workspace
python -m cursus.cli.registry list-steps --workspace {developer_id}

# Check for step conflicts
python -m cursus.cli.registry list-steps --conflicts-only

# Validate registry
python -m cursus.cli.registry validate-registry --workspace {developer_id} --check-conflicts
```

## Best Practices

1. **Unique Step Names**: Use descriptive names that include your domain or framework
2. **Documentation**: Document your custom steps thoroughly
3. **Testing**: Test in workspace context before sharing
4. **Validation**: Regularly validate your registry for consistency

## Support

For questions or issues:
1. Check the [Hybrid Registry Developer Guide](../../slipbox/0_developer_guide/hybrid_registry_guide.md)
2. Validate your setup: `python -m cursus.cli.registry validate-registry --workspace {developer_id}`
3. Contact the development team for assistance
"""

    readme_file.write_text(readme_content)
    return readme_file


def create_example_implementations(workspace_dir: Path, developer_id: str) -> None:
    """Create example step implementations for reference."""
    examples_dir = workspace_dir / "examples"

    # Create example config
    example_config = examples_dir / "example_custom_step_config.py"
    example_config.write_text(
        f'''"""
Example custom step configuration for {developer_id} workspace.
"""
from cursus.core.base.config_base import BasePipelineConfig
from pydantic import Field, ConfigDict
from typing import Optional

class ExampleCustomStepConfig(BasePipelineConfig):
    """Example configuration for custom processing step."""
    
    # Custom parameters
    custom_parameter: str = Field(..., description="Custom processing parameter")
    optional_setting: Optional[bool] = Field(default=True, description="Optional setting")
    
    # Workspace identification
    workspace_id: str = Field(default="{developer_id}", description="Workspace identifier")
    
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )
'''
    )

    # Create example builder
    example_builder = examples_dir / "example_custom_step_builder.py"
    example_builder.write_text(
        f'''"""
Example custom step builder for {developer_id} workspace.
"""
from cursus.core.base.builder_base import StepBuilderBase
from .example_custom_step_config import ExampleCustomStepConfig

class ExampleCustomStepBuilder(StepBuilderBase):
    """Example builder for custom processing step."""
    
    def __init__(self, config: ExampleCustomStepConfig):
        super().__init__(config)
        self.config = config
    
    def build_step(self):
        """Build the custom processing step."""
        # Implementation here
        pass
'''
    )


def validate_workspace_setup(workspace_path: str, developer_id: str) -> None:
    """Validate that workspace setup is correct."""
    workspace_dir = Path(workspace_path)

    # Check required directories exist
    required_dirs = [
        "src/cursus_dev/registry",
        "src/cursus_dev/steps/builders",
        "src/cursus_dev/steps/configs",
        "test",
    ]

    for dir_path in required_dirs:
        full_path = workspace_dir / dir_path
        if not full_path.exists():
            raise ValueError(f"Required directory missing: {dir_path}")

    # Check registry file exists and is valid
    registry_file = workspace_dir / "src/cursus_dev/registry/workspace_registry.py"
    if not registry_file.exists():
        raise ValueError("Registry file not created")

    # Basic validation that registry file can be read
    try:
        content = registry_file.read_text()
        if "LOCAL_STEPS" not in content or "WORKSPACE_METADATA" not in content:
            raise ValueError("Registry file missing required sections")
    except Exception as e:
        raise ValueError(f"Registry validation failed: {e}")


def copy_registry_from_developer(
    workspace_path: str, developer_id: str, source_developer: str
) -> str:
    """Copy registry configuration from existing developer workspace."""
    source_path = Path(
        f"developer_workspaces/developers/{source_developer}/src/cursus_dev/registry/workspace_registry.py"
    )

    if not source_path.exists():
        raise ValueError(f"Source developer '{source_developer}' has no registry file")

    # Read source registry content
    try:
        with open(source_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        raise ValueError(f"Failed to read source registry: {e}")

    # Replace developer ID references in content
    content = content.replace(f'"{source_developer}"', f'"{developer_id}"')
    content = content.replace(f"'{source_developer}'", f"'{developer_id}'")
    content = content.replace(
        f"developer_id: {source_developer}", f"developer_id: {developer_id}"
    )

    # Create target directory and write content
    target_path = Path(workspace_path) / "src/cursus_dev/registry/workspace_registry.py"
    target_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(target_path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception as e:
        raise ValueError(f"Failed to write target registry: {e}")

    return str(target_path)

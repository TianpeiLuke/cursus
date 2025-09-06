# Workspace-Aware Developer Guide: Adding a New Step to the Pipeline

**Version**: 1.0.0  
**Date**: September 5, 2025  
**Author**: Tianpei Xie

## Overview

This guide provides a standardized procedure for adding a new step to the pipeline system using the **workspace-aware isolated project development** approach. This approach is designed for developers working in `development/projects/*/src/cursus_dev/` with hybrid registry access to both shared (`src/cursus`) and project-specific code.

Our pipeline architecture follows the same specification-driven approach with a six-layer design, but implemented in an isolated project environment:

1. **Step Specifications**: Define inputs and outputs with logical names (in `src/cursus_dev/steps/`)
2. **Script Contracts**: Define container paths for script inputs/outputs (in `src/cursus_dev/steps/`)
3. **Processing Scripts**: Implement SageMaker-compatible business logic (in `src/cursus_dev/steps/scripts/`)
4. **Step Builders**: Connect specifications and contracts via SageMaker with workspace-aware registration (in `src/cursus_dev/steps/builders/`)
5. **Configuration Classes**: Define three-tier config structure with field management (in `src/cursus_dev/steps/`)
6. **Hyperparameters**: Define model-specific configuration parameters for training steps (in `src/cursus_dev/steps/`)

Key workspace-aware improvements include:
- **Isolated Development Environment**: Project-specific code in dedicated `src/cursus_dev/` directory
- **Hybrid Registry Access**: Automatic fallback from project-specific to shared components
- **Workspace Context Management**: Project-specific registry context with controlled integration
- **Cross-Workspace Validation**: Comprehensive validation across project and shared code boundaries

## Development Approach: Isolated Project Development

This guide focuses on **Isolated Project Development** in `development/projects/*/` with the following characteristics:

- **Isolated development environment** with project-specific code in `src/cursus_dev/`
- **Hybrid access** to both shared (`src/cursus`) and custom (`src/cursus_dev`) code
- **Project-specific responsibility** with controlled integration points
- **Registry context**: Project-specific workspace with fallback to shared components

### Project Structure
```
development/projects/your_project/
├── src/cursus_dev/              # Project-specific code (isolated from main src/cursus)
│   ├── steps/                   # Project-specific step implementations
│   │   ├── __init__.py
│   │   ├── config_your_step.py  # Project configuration classes
│   │   ├── your_step_contract.py # Project script contracts
│   │   ├── your_step_spec.py    # Project step specifications
│   │   ├── scripts/             # Project processing scripts
│   │   │   └── your_script.py
│   │   └── builders/            # Project step builders
│   │       └── builder_your_step.py
│   ├── configs/                 # Project-specific configurations
│   └── utils/                   # Project utility functions
├── tests/                       # Project-specific tests
├── data/                        # Project data files
└── workspace_config.yaml        # Workspace configuration
```

## Table of Contents

1. [Workspace Setup Guide](ws_workspace_setup_guide.md)
2. [Step Creation Process](ws_creation_process.md)
3. [Detailed Component Guide](ws_component_guide.md)
   - [Script Contract Development](ws_script_contract.md)
   - [Step Specification Development](ws_step_specification.md)
   - [Step Builder Implementation](ws_step_builder.md)
4. [Workspace-Specific Guides](ws_hybrid_registry_integration.md)
   - [Hybrid Registry Integration](ws_hybrid_registry_integration.md)
   - [Shared Code Access Patterns](ws_shared_code_access_patterns.md)
   - [Workspace CLI Reference](ws_workspace_cli_reference.md)
   - [Testing in Isolated Projects](ws_testing_in_isolated_projects.md)
5. [Design Principles](ws_design_principles.md)
6. [Best Practices](ws_best_practices.md)
7. [Standardization Rules](ws_standardization_rules.md)
8. [Validation Checklist](ws_validation_checklist.md)

## Quick Start for Isolated Project Development

### 1. Initialize or Activate Project Workspace

**For New Projects:**
```bash
# Create new isolated project workspace
cursus init-workspace --project your_project --type isolated

# Navigate to project directory
cd development/projects/your_project

# Verify workspace setup
cursus list-workspaces
cursus list-steps --workspace your_project
```

**For Existing Projects:**
```bash
# Navigate to existing project
cd development/projects/your_project

# Activate project workspace
cursus activate-workspace your_project

# Verify current workspace context
cursus current-workspace
```

### 2. Set Workspace Context in Code

```python
from cursus.registry.hybrid.manager import UnifiedRegistryManager

# Get registry instance
registry = UnifiedRegistryManager()

# Set project workspace context
registry.set_workspace_context("your_project")

# Verify workspace context
print(f"Current workspace: {registry.get_current_workspace()}")
```

### 3. Follow the Isolated Project Creation Process

1. Review the [workspace setup guide](ws_workspace_setup_guide.md) to ensure proper project initialization
2. Follow the [step creation process](ws_creation_process.md) to implement all required components in `src/cursus_dev/`
3. **Run workspace-aware validation**:
   ```bash
   # Validate step alignment (4-tier validation)
   cursus validate-alignment --step YourStepType --workspace your_project
   
   # Run comprehensive builder tests
   cursus validate-builder --step YourStepType --workspace your_project
   
   # Validate hybrid registry integration
   cursus validate-registry --workspace your_project
   
   # Check for conflicts with shared code
   cursus check-conflicts --workspace your_project
   ```
4. Validate your implementation using the [workspace-aware validation checklist](ws_validation_checklist.md)

### 4. Integration Testing

```bash
# Test cross-workspace compatibility
cursus test --workspace your_project --include-shared

# Test hybrid registry resolution
cursus test-registry-resolution --workspace your_project

# Validate integration with shared components
cursus validate-integration --workspace your_project
```

## Step Creation Process Overview

The complete step creation process for isolated projects follows this sequence:

1. **Initialize Project Workspace** - Set up isolated development environment
2. **Define Step Specification** - Establish logical inputs/outputs in `src/cursus_dev/steps/`
3. **Create Script Contract** - Define SageMaker container paths in `src/cursus_dev/steps/`
4. **Develop Processing Script** - Implement business logic in `src/cursus_dev/steps/scripts/`
5. **Build Step Builder** - Connect specification and contract in `src/cursus_dev/steps/builders/`
6. **Create Configuration Classes** - Implement three-tier config design in `src/cursus_dev/steps/`
7. **Add Hyperparameters** (if training step) - Define model-specific configuration in `src/cursus_dev/steps/`
8. **Register with Workspace Context** - Register with UnifiedRegistryManager using project workspace
9. **Validate with Workspace Awareness** - Run comprehensive validation including cross-workspace checks
10. **Integration Testing** - Test hybrid registry integration and shared component compatibility

Each step builds upon the previous ones, ensuring alignment across all layers while maintaining isolation from the main codebase.

## Processing Script Development in Isolated Projects

Processing scripts in isolated projects follow the same unified main function interface but are stored in the project-specific directory structure:

### Script Location
```
development/projects/your_project/src/cursus_dev/steps/scripts/your_script.py
```

### Key Requirements

1. **Unified Main Function Interface**: Same standardized signature as main workspace:
   ```python
   def main(
       input_paths: Dict[str, str],
       output_paths: Dict[str, str], 
       environ_vars: Dict[str, str],
       job_args: argparse.Namespace,
       logger=None
   ) -> Any:
       """Main processing function with standardized interface."""
       pass
   ```

2. **SageMaker Container Compatibility**: Same container path requirements
3. **Script Contract Alignment**: Must align with project-specific script contracts
4. **Shared Component Access**: Can import and use shared utilities from `src/cursus/`

### Example Project Script Structure
```python
# File: development/projects/your_project/src/cursus_dev/steps/scripts/your_script.py

import argparse
from typing import Dict, Any
import logging

# Import shared utilities (hybrid access)
from cursus.core.utils import setup_logging
from cursus.core.data_utils import load_data, save_data

# Import project-specific utilities
from cursus_dev.utils.project_helpers import custom_processing

def main(
    input_paths: Dict[str, str],
    output_paths: Dict[str, str], 
    environ_vars: Dict[str, str],
    job_args: argparse.Namespace,
    logger=None
) -> Any:
    """Project-specific processing script with hybrid component access."""
    
    if logger is None:
        logger = setup_logging()  # Using shared utility
    
    logger.info(f"Starting project-specific processing for {job_args.step_type}")
    
    try:
        # Load input data using shared utility
        input_data = load_data(input_paths['input'])
        
        # Apply project-specific processing
        processed_data = custom_processing(input_data, job_args)
        
        # Save output using shared utility
        save_data(processed_data, output_paths['output'])
        
        logger.info("Project-specific processing completed successfully")
        return {"status": "success", "records_processed": len(processed_data)}
        
    except Exception as e:
        logger.error(f"Project-specific processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Standard CLI interface
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-type", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    
    args = parser.parse_args()
    
    input_paths = {"input": args.input_path}
    output_paths = {"output": args.output_path}
    environ_vars = {}
    
    main(input_paths, output_paths, environ_vars, args)
```

## Workspace-Aware Step Builder Registration

Step builders in isolated projects are registered with workspace context:

### Project-Specific Registration
```python
# File: development/projects/your_project/src/cursus_dev/steps/builders/builder_your_step.py

from cursus.registry.hybrid.manager import UnifiedRegistryManager
from cursus.core.step_builder import StepBuilderBase

class YourStepStepBuilder(StepBuilderBase):
    """Project-specific step builder with hybrid registry integration."""
    
    def __init__(self):
        super().__init__()
        # Access shared components as needed
        self.shared_validator = self._get_shared_validator()
    
    def _get_shared_validator(self):
        """Access shared validation components via hybrid registry."""
        registry = UnifiedRegistryManager()
        # Temporarily switch to main workspace to access shared components
        original_workspace = registry.get_current_workspace()
        registry.set_workspace_context("main")
        validator = registry.get_component("validation_utils")
        registry.set_workspace_context(original_workspace)
        return validator
    
    def build_step(self, config):
        """Build step with project-specific logic and shared component access."""
        # Implementation here
        pass

# Register with project workspace context
registry = UnifiedRegistryManager()
registry.set_workspace_context("your_project")
registry.register_step_builder("YourStepType", YourStepStepBuilder)
```

### Legacy Decorator Support with Workspace Context
```python
from cursus.registry.decorators import register_builder

# Set workspace context before using decorator
registry = UnifiedRegistryManager()
registry.set_workspace_context("your_project")

@register_builder("YourStepType")
class YourStepStepBuilder(StepBuilderBase):
    """Project-specific builder registered with workspace context."""
    # Implementation here
```

## Hybrid Registry Integration Patterns

### Accessing Shared Components
```python
from cursus.registry.hybrid.manager import UnifiedRegistryManager

# Get registry instance
registry = UnifiedRegistryManager()

# Set project workspace context
registry.set_workspace_context("your_project")

# Access shared step (automatically falls back to main workspace)
shared_step = registry.get_step_builder("preprocessing_step")

# Access project-specific step (from current workspace)
project_step = registry.get_step_builder("your_project_custom_step")
```

### Resolution Priority
1. **Project-specific code** (`src/cursus_dev`) - First priority
2. **Shared code** (`src/cursus`) - Fallback
3. **Default implementations** - Final fallback

### Workspace Context Management
```python
# Save current workspace context
original_workspace = registry.get_current_workspace()

# Temporarily switch to access shared components
registry.set_workspace_context("main")
shared_component = registry.get_component("shared_utility")

# Restore original workspace context
registry.set_workspace_context(original_workspace)
```

## Daily Development Workflow

### 1. Workspace Activation
```bash
cd development/projects/your_project
cursus activate-workspace your_project
```

### 2. Development Cycle
- Edit code in `src/cursus_dev/`
- Access shared utilities from `src/cursus/` as needed
- Test with workspace-aware registry
- Validate integration with shared components

### 3. Testing and Validation
```bash
# Run project-specific tests
cursus test --workspace your_project

# Validate registry integration
cursus validate-registry --workspace your_project

# Check for conflicts with shared code
cursus check-conflicts --workspace your_project

# Run cross-workspace integration tests
cursus test-integration --workspace your_project --include-shared
```

### 4. Integration Preparation
```bash
# Validate alignment with shared components
cursus validate-alignment --workspace your_project --check-shared

# Generate integration report
cursus generate-integration-report --workspace your_project

# Test deployment readiness
cursus test-deployment --workspace your_project
```

## Key Differences from Main Workspace Development

### Isolated Project Development (`development/projects/*/`)
- **Project-specific code location**: `src/cursus_dev/` directory
- **Hybrid component access**: Controlled access to shared components via registry
- **Workspace context management**: Explicit workspace context setting required
- **Isolated testing**: Project-specific tests with optional shared component integration
- **Controlled integration**: Integration checkpoints before moving to production

### Main Workspace Development (`src/cursus/`)
- **Direct code location**: `src/cursus/` directory
- **Direct component access**: Immediate access to all existing components
- **Default workspace context**: "main" workspace context
- **Integrated testing**: Full system testing by default
- **Immediate integration**: Changes immediately affect entire system

## Additional Resources

- [Workspace Setup Guide](ws_workspace_setup_guide.md) - Detailed project initialization
- [Hybrid Registry Integration](ws_hybrid_registry_integration.md) - Registry usage patterns
- [Shared Code Access Patterns](ws_shared_code_access_patterns.md) - Accessing shared components
- [Testing in Isolated Projects](ws_testing_in_isolated_projects.md) - Testing strategies
- [Workspace CLI Reference](ws_workspace_cli_reference.md) - CLI tools and commands

## Related Documentation

- [Creation Process (Workspace-Aware)](ws_creation_process.md) - Complete step-by-step workflow
- [Component Guide (Workspace-Aware)](ws_component_guide.md) - 6-layer architecture in isolated context
- [Step Builder Registry Guide (Workspace-Aware)](ws_step_builder_registry_guide.md) - Registry system usage
- [Best Practices (Workspace-Aware)](ws_best_practices.md) - Development best practices
- [Validation Checklist (Workspace-Aware)](ws_validation_checklist.md) - Quality assurance checklist

### Main Developer Guide References
- [Main Developer Guide](../0_developer_guide/README.md) - Traditional main workspace development
- [Adding a New Pipeline Step](../0_developer_guide/adding_new_pipeline_step.md) - Main workspace step creation
- [Step Builder Registry Guide](../0_developer_guide/step_builder_registry_guide.md) - Main workspace registry usage

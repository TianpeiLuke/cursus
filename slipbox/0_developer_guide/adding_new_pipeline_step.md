# Developer Guide: Adding a New Step to the Pipeline

**Version**: 1.2.3  
**Date**: September 6, 2025  
**Author**: Tianpei Xie

## Overview

This guide provides a standardized procedure for adding a new step to the pipeline system using the modern workspace-aware development approach. Following these guidelines ensures that your implementation maintains consistency with the existing code structure and adheres to our design principles.

Our pipeline architecture follows a specification-driven approach with a six-layer design:

1. **Step Specifications**: Define inputs and outputs with logical names
2. **Script Contracts**: Define container paths for script inputs/outputs
3. **Processing Scripts**: Implement SageMaker-compatible business logic with unified interface
4. **Step Builders**: Connect specifications and contracts via SageMaker with workspace-aware registration
5. **Configuration Classes**: Define three-tier config structure with field management
6. **Hyperparameters**: Define model-specific configuration parameters (for training steps)

Key architectural improvements include:
- **Three-Tier Config Classification**: Clear separation of user inputs, system defaults, and derived values
- **Workspace-Aware Registry**: UnifiedRegistryManager with workspace context management
- **CLI Integration**: Automated workspace management and validation commands
- **4-Tier Alignment Validation**: Comprehensive validation framework ensuring system integrity

## Development Approaches

This guide supports two development approaches:

### Main Workspace Development (`src/cursus/`)
- **Direct development** in the main codebase
- **Immediate integration** with existing components
- **Shared responsibility** for code quality
- **Registry context**: Default "main" workspace

### Isolated Project Development (`development/projects/*/`)
- **Isolated development environment** with project-specific code in `src/cursus_dev/`
- **Hybrid access** to both shared (`src/cursus`) and custom (`src/cursus_dev`) code
- **Project-specific responsibility** with controlled integration
- **Registry context**: Project-specific workspace with fallback to shared components

Choose the approach that best fits your development needs. This guide covers both patterns.

## Table of Contents

1. [Prerequisites](prerequisites.md)
2. [Step Creation Process](creation_process.md)
3. [Detailed Component Guide](component_guide.md)
   - [Script Contract Development](script_contract.md)
   - [Step Specification Development](step_specification.md)
   - [Processing Script Development](script_development_guide.md)
   - [Step Builder Implementation](step_builder.md)
   - [Configuration Classes Development](three_tier_config_design.md)
   - [Config Field Manager Guide](config_field_manager_guide.md)
   - [Adding a New Hyperparameter Class](hyperparameter_class.md)
   - [Step Builder Registry Guide](step_builder_registry_guide.md)
4. [Design Principles](design_principles.md)
5. [Best Practices](best_practices.md)
6. [Standardization Rules](standardization_rules.md)
7. [Common Pitfalls to Avoid](common_pitfalls.md)
8. [Alignment Rules](alignment_rules.md)
9. [Validation Framework Guide](validation_framework_guide.md)
10. [Example](example.md)
11. [Validation Checklist](validation_checklist.md)

## Quick Start

### For Main Workspace Development (`src/cursus/`)

To add a new step to the main workspace:

1. **Set workspace context** (if needed for testing):
   ```bash
   # CLI approach
   cursus set-workspace main
   ```

   # Or in Python code
   ```python
   from cursus.registry.hybrid.manager import UnifiedRegistryManager
   registry = UnifiedRegistryManager()
   registry.set_workspace_context("main")
   ```

2. Review the [prerequisites](prerequisites.md) to ensure you have all required information

3. Follow the [step creation process](creation_process.md) to implement all required components

4. **Run validation framework tests** with workspace awareness:
   ```bash
   # Validate step alignment (4-tier validation)
   cursus validate-alignment --step YourStepType --workspace main
   
   # Run comprehensive builder tests
   cursus validate-builder --step YourStepType --workspace main
   
   # Validate registry integration
   cursus validate-registry --workspace main
   ```

5. Validate your implementation using the [validation checklist](validation_checklist.md)

### For Isolated Project Development (`development/projects/*/`)

To add a new step in an isolated project:

1. **Initialize or activate project workspace**:
   ```bash
   # Create new project (if needed)
   cursus init-workspace --project your_project --type isolated
   
   # Activate existing project
   cd development/projects/your_project
   cursus activate-workspace your_project
   ```

2. **Set up project structure** (if new project):
   ```bash
   # Verify project structure
   cursus list-workspaces
   cursus list-steps --workspace your_project
   ```

3. **Develop in isolated environment**:
   - Create step components in `src/cursus_dev/steps/`
   - Access shared components from `src/cursus/` as needed
   - Use workspace-aware registry for registration

4. **Run workspace-aware validation**:
   ```bash
   # Validate project-specific steps
   cursus validate-alignment --step YourStepType --workspace your_project
   
   # Test hybrid registry integration
   cursus validate-registry --workspace your_project
   
   # Check for conflicts with shared code
   cursus check-conflicts --workspace your_project
   ```

5. **Integration testing**:
   ```bash
   # Test cross-workspace compatibility
   cursus test --workspace your_project --include-shared
   ```

For detailed guidance on specific components, refer to the relevant sections in the [detailed component guide](component_guide.md).

## Step Creation Process Overview

The complete step creation process follows this sequence:

1. **Define Step Specification** - Establish logical inputs/outputs and step interface
2. **Create Script Contract** - Define SageMaker container paths and script interface
3. **Develop Processing Script** - Implement SageMaker-compatible business logic
4. **Build Step Builder** - Connect specification and contract via SageMaker integration
5. **Create Configuration Classes** - Implement three-tier config design with field management
6. **Add Hyperparameters** (if training step) - Define model-specific configuration
7. **Register and Validate** - Register with UnifiedRegistryManager and run validation tests

Each step builds upon the previous ones, ensuring alignment across all layers of the architecture.

## Processing Script Development

Processing scripts are the core business logic that runs within SageMaker containers. They must follow standardized patterns for testability, container compatibility, and alignment with script contracts.

### Key Requirements

1. **Unified Main Function Interface**: All scripts must implement a standardized main function signature:
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

2. **SageMaker Container Compatibility**: Scripts must work with SageMaker's standard container paths:
   - Processing: `/opt/ml/processing/input/` and `/opt/ml/processing/output/`
   - Training: `/opt/ml/input/data/`, `/opt/ml/model/`, `/opt/ml/output/data/`
   - Transform: `/opt/ml/input/data/` and `/opt/ml/output/`

3. **Script Contract Alignment**: Scripts must match the expectations defined in their script contracts, including:
   - Entry point filename
   - Expected input/output paths
   - CLI arguments (contract uses hyphens, scripts use underscores via argparse)
   - Environment variables

### Development Workflow

1. **Choose Script Template**: Select appropriate template (processing, training, or transform)
2. **Implement Business Logic**: Add your processing logic to the main function
3. **Add Error Handling**: Include comprehensive error handling and logging
4. **Create Success/Failure Markers**: Ensure proper completion signaling
5. **Unit Test**: Test the main function with various inputs
6. **Integration Test**: Test in simulated container environment

### Script Storage Locations

**Shared Workspace (Traditional)**:
```
src/cursus/steps/scripts/your_script.py
```

**Isolated Development Workspace (Recommended)**:
```
development/projects/project_name/src/cursus_dev/steps/scripts/your_script.py
```

For comprehensive guidance including templates, best practices, and testing approaches, see the [Script Development Guide](script_development_guide.md).

## Adding a New Hyperparameter Class

When adding a new training step, you will likely need to create a custom hyperparameter class that inherits from the base `ModelHyperparameters` class.

For detailed guidance, see the [Adding a New Hyperparameter Class](hyperparameter_class.md) guide, which covers:

- Creating the hyperparameter class file
- Registering the class in the hyperparameter registry
- Integrating with training config classes
- Setting up training scripts to use hyperparameters
- Configuring step builders to pass hyperparameters to SageMaker
- Testing your hyperparameter implementation

## Three-Tier Config Design

All step configurations should follow our Three-Tier Config Design pattern, which provides clear separation between different types of configuration fields:

- **Tier 1 (Essential Fields)**: Required inputs explicitly provided by users
- **Tier 2 (System Fields)**: Default values that can be overridden by users
- **Tier 3 (Derived Fields)**: Values calculated from other fields

For implementation details, see the [Three-Tier Config Design](three_tier_config_design.md) guide.

## Step Builder Registry and Workspace-Aware Registration

Step builders are registered with our UnifiedRegistryManager system, which provides workspace-aware caching and context management:

### Main Workspace Registration
```python
from cursus.registry.hybrid.manager import UnifiedRegistryManager

# Get registry instance
registry = UnifiedRegistryManager()

# Set workspace context (defaults to "main")
registry.set_workspace_context("main")

# Register step builder
registry.register_step_builder("YourStepType", YourStepStepBuilder)
```

### Isolated Project Registration
```python
from cursus.registry.hybrid.manager import UnifiedRegistryManager

# Get registry instance
registry = UnifiedRegistryManager()

# Set project workspace context
registry.set_workspace_context("your_project")

# Register project-specific step builder
registry.register_step_builder("YourStepType", YourStepStepBuilder)
```

### Legacy Decorator Support
The `@register_builder` decorator is still supported for backward compatibility but uses UnifiedRegistryManager internally:

```python
@register_builder("YourStepType")
class YourStepStepBuilder(StepBuilderBase):
    """Builder for your step."""
    # Implementation here
```

For detailed guidance on using the registry system, workspace context management, and resolution strategies, see the [Step Builder Registry Guide](step_builder_registry_guide.md).

## Additional Resources

- [Specification-Driven Architecture](../pipeline_design/specification_driven_design.md)
- [Hybrid Design](../pipeline_design/hybrid_design.md)
- [Script-Specification Alignment](../project_planning/script_specification_alignment_prevention_plan.md)
- [Script Contract](../pipeline_design/script_contract.md)
- [Step Specification](../pipeline_design/step_specification.md)

## Related Documentation

- [Creation Process](creation_process.md) - Complete step-by-step pipeline creation workflow
- [Script Development Guide](script_development_guide.md) - Comprehensive guide for developing SageMaker-compatible scripts
- [Pipeline Catalog Integration Guide](pipeline_catalog_integration_guide.md) - How to integrate your pipeline steps with the Zettelkasten-inspired catalog system
- [Step Builder Registry Guide](step_builder_registry_guide.md) - Comprehensive guide to the UnifiedRegistryManager and hybrid registry system
- [Step Builder Registry Usage](step_builder_registry_usage.md) - Practical examples and usage patterns for registry operations
- [Step Builder](step_builder.md) - Detailed guide to step builder implementation
- [Step Specification](step_specification.md) - How to define step specifications
- [Validation Framework Guide](validation_framework_guide.md) - Comprehensive validation system
- [Best Practices](best_practices.md) - Development best practices and guidelines

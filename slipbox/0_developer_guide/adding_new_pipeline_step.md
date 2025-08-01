# Developer Guide: Adding a New Step to the Pipeline

**Version**: 2.0  
**Date**: July 12, 2025  
**Author**: MODS Development Team

## Overview

This guide provides a standardized procedure for adding a new step to the pipeline system. Following these guidelines ensures that your implementation maintains consistency with the existing code structure and adheres to our design principles.

Our pipeline architecture follows a specification-driven approach with a four-layer design:

1. **Step Specifications**: Define inputs and outputs with logical names
2. **Script Contracts**: Define container paths for script inputs/outputs
3. **Step Builders**: Connect specifications and contracts via SageMaker with auto-registration
4. **Processing Scripts**: Implement the actual business logic
5. **Hyperparameters**: Define model-specific configuration parameters (for training steps)

Key architectural improvements include:
- **Three-Tier Config Classification**: Clear separation of user inputs, system defaults, and derived values
- **Auto-Discovery Registry**: Automatic registration of step builders via decorators

## Table of Contents

1. [Prerequisites](prerequisites.md)
2. [Step Creation Process](creation_process.md)
3. [Detailed Component Guide](component_guide.md)
   - [Script Contract Development](script_contract.md)
   - [Step Specification Development](step_specification.md)
   - [Step Builder Implementation](step_builder.md)
   - [Adding a New Hyperparameter Class](hyperparameter_class.md)
   - [Three-Tier Config Design](three_tier_config_design.md)
   - [Step Builder Registry Guide](step_builder_registry_guide.md)
4. [Design Principles](design_principles.md)
5. [Best Practices](best_practices.md)
6. [Standardization Rules](standardization_rules.md)
7. [Common Pitfalls to Avoid](common_pitfalls.md)
8. [Alignment Rules](alignment_rules.md)
9. [Example](example.md)
10. [Validation Checklist](validation_checklist.md)

## Quick Start

To add a new step to the pipeline:

1. Review the [prerequisites](prerequisites.md) to ensure you have all required information
2. Follow the [step creation process](creation_process.md) to implement all required components
3. Validate your implementation using the [validation checklist](validation_checklist.md)

For detailed guidance on specific components, refer to the relevant sections in the [detailed component guide](component_guide.md).

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

## Step Builder Registry and Auto-Discovery

Step builders are automatically discovered and registered with our registry system using the `@register_builder` decorator:

```python
@register_builder("YourStepType")
class YourStepStepBuilder(StepBuilderBase):
    """Builder for your step."""
    # Implementation here
```

For detailed guidance on using the registry, see the [Step Builder Registry Guide](step_builder_registry_guide.md).

## Additional Resources

- [Specification-Driven Architecture](../pipeline_design/specification_driven_design.md)
- [Hybrid Design](../pipeline_design/hybrid_design.md)
- [Script-Specification Alignment](../project_planning/script_specification_alignment_prevention_plan.md)
- [Script Contract](../pipeline_design/script_contract.md)
- [Step Specification](../pipeline_design/step_specification.md)

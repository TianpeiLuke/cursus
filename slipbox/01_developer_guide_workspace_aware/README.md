# Workspace-Aware Pipeline Step Developer Guide

This directory contains comprehensive documentation for developing new steps in isolated project environments using our workspace-aware development approach. This guide is specifically designed for developers working in `development/projects/*/src/cursus_dev/` with hybrid registry access to both shared (`src/cursus`) and project-specific code.

## Overview

The workspace-aware developer guide supports **Isolated Project Development**, where developers work in dedicated project environments with:

- **Isolated development environment** with project-specific code in `src/cursus_dev/`
- **Hybrid access** to both shared (`src/cursus`) and custom (`src/cursus_dev`) code
- **Project-specific responsibility** with controlled integration points
- **Workspace-aware testing** with registry context management
- **Registry context**: Project-specific workspace with fallback to shared components

## Recent Updates (September 2025)

🆕 **New Workspace-Aware Documentation**: This guide has been created to complement the main developer guide with workspace-specific patterns:

- **UnifiedRegistryManager Integration**: Workspace-aware registry usage with project context
- **Isolated Project Development**: Complete workflow for `development/projects/*/` structure
- **Hybrid Code Access**: Patterns for accessing both shared and project-specific components
- **Workspace CLI Integration**: Project management commands and validation tools
- **Cross-Workspace Testing**: Testing strategies for isolated development environments

## Guide Structure

The workspace-aware developer guide mirrors the main developer guide structure while focusing on isolated project development:

### Main Documentation

- **[Adding a New Pipeline Step (Workspace-Aware)](ws_adding_new_pipeline_step.md)** - The main entry point for workspace-aware step development with isolated project focus

### Process Documentation

- **[Workspace Setup Guide](ws_workspace_setup_guide.md)** - Project initialization and workspace configuration
- **[Creation Process (Workspace-Aware)](ws_creation_process.md)** - Complete 10-step process for isolated project development

### Component Documentation

- **[Component Guide (Workspace-Aware)](ws_component_guide.md)** - 6-layer architecture in isolated project context
- **[Step Builder Implementation (Workspace-Aware)](ws_step_builder.md)** - Step builders with workspace-aware registry integration
- **[Step Builder Registry Guide (Workspace-Aware)](ws_step_builder_registry_guide.md)** - Registry usage in project workspace context
- **[Step Builder Registry Usage (Workspace-Aware)](ws_step_builder_registry_usage.md)** - Practical examples for project-specific registry operations
- **[Script Contract Development (Workspace-Aware)](ws_script_contract.md)** - Script contracts in isolated project environment
- **[Step Specification Development (Workspace-Aware)](ws_step_specification.md)** - Step specifications for project-specific steps

### Design and Standards

- **[Design Principles (Workspace-Aware)](ws_design_principles.md)** - Design principles for isolated project development
- **[Best Practices (Workspace-Aware)](ws_best_practices.md)** - Best practices for workspace-aware development
- **[Standardization Rules (Workspace-Aware)](ws_standardization_rules.md)** - Code standards for isolated projects
- **[Validation Checklist (Workspace-Aware)](ws_validation_checklist.md)** - Quality assurance for workspace projects

### Workspace-Specific Documentation

- **[Hybrid Registry Integration](ws_hybrid_registry_integration.md)** - Registry system usage patterns for isolated projects
- **[Shared Code Access Patterns](ws_shared_code_access_patterns.md)** - Accessing `src/cursus` components from isolated projects
- **[Workspace CLI Reference](ws_workspace_cli_reference.md)** - CLI tools and commands for project management
- **[Testing in Isolated Projects](ws_testing_in_isolated_projects.md)** - Testing strategies for workspace-aware development
- **[Deployment and Integration](ws_deployment_and_integration.md)** - Moving code from isolated projects to production

### Advanced Topics

- **[Multi-Project Collaboration](ws_multi_project_collaboration.md)** - Working across multiple isolated projects
- **[Workspace Migration Guide](ws_workspace_migration_guide.md)** - Moving between workspaces and integration strategies
- **[Troubleshooting Workspace Issues](ws_troubleshooting_workspace_issues.md)** - Common problems and solutions for isolated development
- **[Performance Optimization](ws_performance_optimization.md)** - Workspace-specific performance considerations

## Quick Start Summary

**New to workspace-aware development?** Follow this rapid orientation:

1. **Initialize Project Workspace**: Use `cursus init-workspace --project your_project --type isolated`
2. **Understand Isolated Architecture**: Same 6-layer system but with project-specific code in `src/cursus_dev/`
3. **Set Up Development Environment**: Configure workspace context and hybrid registry access
4. **Key Decision Points**:
   - What project-specific functionality do you need?
   - Which shared components from `src/cursus` will you use?
   - How will your project-specific code integrate with shared systems?
   - What testing strategy fits your isolated development needs?
5. **Essential Project Files to Create**:
   - `src/cursus_dev/steps/config_your_step.py` (project-specific configuration)
   - `src/cursus_dev/steps/your_step_contract.py` (project script contract)
   - `src/cursus_dev/steps/scripts/your_script.py` (project processing script)
   - `src/cursus_dev/steps/your_step_spec.py` (project step specification)
   - `src/cursus_dev/steps/builders/builder_your_step.py` (project step builder)
6. **Workspace Validation**: Use workspace-aware validation with `cursus validate-alignment --workspace your_project`
7. **Integration Testing**: Test hybrid registry integration and cross-workspace compatibility

**Experienced with main workspace development?** Jump to [Workspace Setup Guide](ws_workspace_setup_guide.md) to understand the differences and get started quickly.

## Recommended Reading Order

For developers new to workspace-aware development, we recommend this reading order:

1. Start with **[Workspace Setup Guide](ws_workspace_setup_guide.md)** to initialize your project environment
2. Read **[Adding a New Pipeline Step (Workspace-Aware)](ws_adding_new_pipeline_step.md)** for an overview of isolated project development
3. Review the **[Creation Process (Workspace-Aware)](ws_creation_process.md)** for the complete workflow
4. Study **[Hybrid Registry Integration](ws_hybrid_registry_integration.md)** to understand registry usage patterns
5. Learn **[Shared Code Access Patterns](ws_shared_code_access_patterns.md)** for accessing `src/cursus` components
6. Dive deeper into component documentation:
   - **[Component Guide (Workspace-Aware)](ws_component_guide.md)**
   - **[Step Builder Implementation (Workspace-Aware)](ws_step_builder.md)**
   - **[Step Builder Registry Guide (Workspace-Aware)](ws_step_builder_registry_guide.md)**
7. Master workspace-specific tools:
   - **[Workspace CLI Reference](ws_workspace_cli_reference.md)**
   - **[Testing in Isolated Projects](ws_testing_in_isolated_projects.md)**
8. Review design and standards:
   - **[Design Principles (Workspace-Aware)](ws_design_principles.md)**
   - **[Best Practices (Workspace-Aware)](ws_best_practices.md)**
   - **[Standardization Rules (Workspace-Aware)](ws_standardization_rules.md)**
9. Use the **[Validation Checklist (Workspace-Aware)](ws_validation_checklist.md)** to verify your implementation
10. Explore advanced topics as needed for your specific project requirements

## Key Architectural Concepts for Isolated Projects

Our workspace-aware architecture maintains the same 6-layer design but with project-specific implementation:

1. **Step Specifications**: Project-specific definitions in `src/cursus_dev/steps/`
2. **Script Contracts**: Project container paths and environment variables
3. **Processing Scripts**: Project business logic in `src/cursus_dev/steps/scripts/`
4. **Step Builders**: Project builders with hybrid registry integration
5. **Configuration Classes**: Project-specific config with shared component access
6. **Hyperparameters**: Project ML parameters with shared base classes

**Key Workspace-Aware Features**:
- **Hybrid Registry Access**: Automatic fallback from project-specific to shared components
- **Workspace Context Management**: Isolated development with controlled integration points
- **Cross-Workspace Testing**: Validation across project and shared code boundaries
- **CLI Workspace Management**: Automated project setup, validation, and integration tools

## Development Workflow Comparison

### Traditional Main Workspace (`src/cursus/`)
- Direct modification of shared codebase
- Immediate integration with all existing components
- Shared responsibility for system stability
- Global impact of changes

### Isolated Project Workspace (`development/projects/*/`)
- Project-specific code in dedicated `src/cursus_dev/` directory
- Controlled access to shared components via hybrid registry
- Project-specific responsibility with integration checkpoints
- Isolated development with optional integration

## Project Structure Overview

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
│   ├── test_your_steps.py
│   └── integration/
├── data/                        # Project data files
└── workspace_config.yaml        # Workspace configuration
```

## Getting Help

If you encounter issues or have questions while developing in isolated projects:

1. Consult the **[Troubleshooting Workspace Issues](ws_troubleshooting_workspace_issues.md)** document
2. Use the **[Validation Checklist (Workspace-Aware)](ws_validation_checklist.md)** to identify potential issues
3. Review workspace-specific CLI commands in **[Workspace CLI Reference](ws_workspace_cli_reference.md)**
4. Check hybrid registry integration patterns in **[Hybrid Registry Integration](ws_hybrid_registry_integration.md)**
5. Reach out to the architecture team for workspace-specific assistance

## Contributing to the Workspace-Aware Guide

If you identify gaps in the workspace-aware documentation or have suggestions for improvements:

1. Document your proposed changes with workspace context
2. Test changes in isolated project environment
3. Discuss with the architecture team
4. Update the relevant workspace-aware documentation
5. Ensure consistency with main developer guide where applicable

The workspace-aware developer guide evolves with our isolated project development practices and hybrid registry system.

## Related Documentation

### Main Developer Guide
- [Main Developer Guide](../0_developer_guide/README.md) - Traditional main workspace development
- [Adding a New Pipeline Step](../0_developer_guide/adding_new_pipeline_step.md) - Main workspace step creation

### Design Documents
- [Workspace-Aware System Design](../1_design/workspace_aware_system_master_design.md) - System architecture
- [Hybrid Registry Design](../1_design/hybrid_registry_standardization_enforcement_design.md) - Registry system design

### Project Planning
- [Workspace-Aware Documentation Plan](../2_project_planning/2025-09-05_developer_guide_workspace_aware_documentation_plan.md) - This documentation project plan

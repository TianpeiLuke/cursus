# Pipeline Step Developer Guide

This directory contains comprehensive documentation for developing new steps in our SageMaker-based ML pipeline architecture. The guide is designed to help both new and experienced developers create pipeline steps that align with our modern architectural patterns and best practices.

## Recent Updates (September 2025)

🆕 **Major Documentation Modernization**: The developer guide has been comprehensively updated to reflect the latest architectural changes:

- **UnifiedRegistryManager System**: Updated from legacy registry patterns to modern hybrid registry system with workspace-aware development
- **6-Layer Architecture**: Enhanced from 4-layer to 6-layer system including Script Development and Configuration Classes
- **Script Development Integration**: Added comprehensive script development guide with unified main function interface
- **Pipeline Catalog Integration**: New Zettelkasten-inspired pipeline catalog with connection-based discovery
- **Workspace-Aware Validation**: Enhanced validation framework with workspace isolation capabilities
- **Consistent Documentation Structure**: All guides now feature consistent numbering and cross-references

## Guide Structure

The developer guide is organized into several interconnected documents:

### Main Documentation

- **[Adding a New Pipeline Step](adding_new_pipeline_step.md)** - The main entry point providing an overview of the step development process with 6-layer architecture

### Process Documentation

- **[Prerequisites](prerequisites.md)** - What you need before starting development (updated for modern system)
- **[Creation Process](creation_process.md)** - Complete 10-step process for adding a new pipeline step with consistent numbering

### Component Documentation

- **[Component Guide](component_guide.md)** - Overview of the 6-layer architecture and component relationships
- **[Script Development Guide](script_development_guide.md)** - 🆕 **NEW**: Comprehensive guide for developing processing scripts with unified main function interface
- **[Script Contract Development](script_contract.md)** - Detailed guide for developing script contracts
- **[Step Specification Development](step_specification.md)** - Detailed guide for developing step specifications
- **[Step Builder Implementation](step_builder.md)** - Updated guide for implementing step builders with UnifiedRegistryManager

### Registry and Validation

- **[Step Builder Registry Guide](step_builder_registry_guide.md)** - Comprehensive guide to the UnifiedRegistryManager and hybrid registry system
- **[Step Builder Registry Usage](step_builder_registry_usage.md)** - Practical examples and usage patterns for registry operations
- **[Validation Framework Guide](validation_framework_guide.md)** - Updated comprehensive validation usage instructions with workspace-aware system

### Pipeline Integration

- **[Pipeline Catalog Integration Guide](pipeline_catalog_integration_guide.md)** - 🆕 **NEW**: Complete guide for integrating with the Zettelkasten-inspired pipeline catalog system
- **[Three-Tier Config Design](three_tier_config_design.md)** - Configuration design patterns with Essential/System/Derived field classification

### Best Practices and Guidelines

- **[Design Principles](design_principles.md)** - Core design principles to follow
- **[Best Practices](best_practices.md)** - Recommended best practices for development
- **[Standardization Rules](standardization_rules.md)** - Updated architectural constraints for modern system patterns
- **[Common Pitfalls](common_pitfalls.md)** - Common mistakes to avoid
- **[Alignment Rules](alignment_rules.md)** - Updated alignment guidance with workspace-aware validation examples
- **[Validation Checklist](validation_checklist.md)** - Comprehensive checklist for validating implementations

### Examples

- **[Example Implementation](example.md)** - Complete example of adding a new pipeline step

## Quick Start Summary

**New to pipeline development?** Follow this rapid orientation:

1. **Understand the Architecture**: 6-layer system - Step Specifications → Script Contracts → Processing Scripts → Step Builders → Configuration Classes → Hyperparameters
2. **Check Prerequisites**: Ensure you have step requirements and understand the business logic
3. **Follow the Process**: Set up workspace → Create config → Develop contract → Create script → Build specification → Implement builder → Register → Validate → Test → Integrate
4. **Key Decision Points**:
   - What inputs/outputs does your step need?
   - What SageMaker step type (Processing, Training, Transform)?
   - What job type variants (training, calibration, validation)?
   - Which development approach (main workspace vs isolated project)?
5. **Essential Files to Create**:
   - `config_your_step.py` (configuration with three-tier design)
   - `your_step_contract.py` (script contract)
   - `your_script.py` (processing script with unified main function)
   - `your_step_spec.py` (step specification)
   - `builder_your_step.py` (step builder)
6. **Validation**: Use 4-tier alignment validation and Universal Step Builder Test before integration
7. **Integration**: Connect with Pipeline Catalog using Zettelkasten metadata

**Experienced developers?** Jump to [Creation Process](creation_process.md) for the complete 10-step procedure.

## Recommended Reading Order

For new developers, we recommend the following reading order:

1. Start with **[Adding a New Pipeline Step](adding_new_pipeline_step.md)** for an overview of the 6-layer architecture
2. Check **[Prerequisites](prerequisites.md)** to ensure you have everything needed (updated for modern system)
3. Review the **[Creation Process](creation_process.md)** for the complete 10-step procedure with consistent numbering
4. Read the **[Component Guide](component_guide.md)** to understand the 6-layer architecture and component relationships
5. Study the **[Script Development Guide](script_development_guide.md)** for unified main function interface and SageMaker compatibility
6. Dive deeper into specific component documentation:
   - **[Script Contract Development](script_contract.md)**
   - **[Step Specification Development](step_specification.md)**
   - **[Step Builder Implementation](step_builder.md)** (updated for UnifiedRegistryManager)
7. Learn about registry and validation:
   - **[Step Builder Registry Guide](step_builder_registry_guide.md)** for hybrid registry system
   - **[Validation Framework Guide](validation_framework_guide.md)** for workspace-aware validation
8. Understand pipeline integration:
   - **[Pipeline Catalog Integration Guide](pipeline_catalog_integration_guide.md)** for Zettelkasten-based catalog
   - **[Three-Tier Config Design](three_tier_config_design.md)** for configuration patterns
9. Study the **[Example Implementation](example.md)** to see how everything fits together
10. Review best practices and guidelines:
    - **[Design Principles](design_principles.md)**
    - **[Best Practices](best_practices.md)**
    - **[Standardization Rules](standardization_rules.md)** (updated for modern patterns)
    - **[Alignment Rules](alignment_rules.md)** (updated with workspace-aware examples)
    - **[Common Pitfalls](common_pitfalls.md)**
11. Use the **[Validation Checklist](validation_checklist.md)** to verify your implementation

## Key Architectural Concepts

Our pipeline architecture follows a specification-driven approach with a modern 6-layer design:

1. **Step Specifications**: Define inputs and outputs with logical names and dependency relationships
2. **Script Contracts**: Define container paths and environment variables for script execution
3. **Processing Scripts**: Implement business logic using unified main function interface for testability
4. **Step Builders**: Connect specifications and contracts via SageMaker with UnifiedRegistryManager integration
5. **Configuration Classes**: Manage step parameters using three-tier field classification (Essential/System/Derived)
6. **Hyperparameters**: Handle ML-specific parameter tuning and optimization

**Key Modern Features**:
- **UnifiedRegistryManager**: Single consolidated registry replacing legacy patterns
- **Workspace-Aware Development**: Support for both traditional and isolated development approaches
- **4-Tier Validation**: Script-Contract, Contract-Specification, Specification-Dependencies, Builder-Configuration alignment
- **Pipeline Catalog Integration**: Zettelkasten-inspired discovery with connection-based relationships

Understanding these layers and their relationships is crucial for successful step development in the modern architecture.

## Using AI to Assist Development

For guidance on using Claude v3 to assist with pipeline step development, see the AI prompt templates in the [../developer_prompts](../developer_prompts) directory.

## Getting Help

If you encounter issues or have questions while developing a new pipeline step:

1. Consult the **[Common Pitfalls](common_pitfalls.md)** document
2. Use the **[Validation Checklist](validation_checklist.md)** to identify potential issues
3. Review the **[Example Implementation](example.md)** for reference
4. Reach out to the architecture team for assistance

## Contributing to the Guide

If you identify gaps in the documentation or have suggestions for improvements:

1. Document your proposed changes
2. Discuss with the architecture team
3. Update the relevant documentation
4. Ensure consistency across all documents

The developer guide is a living document that evolves with our architecture.

---
tags:
  - entry_point
  - code_structure
  - architecture
  - index
  - documentation_map
keywords:
  - code organization
  - component index
  - design documentation
  - architecture reference
topics:
  - code structure
  - system components
  - documentation mapping
language: python
date of note: 2025-11-09
---

# Cursus Code Structure & Documentation Index

## Overview

This index card maps the Cursus codebase structure to related design documentation, providing a navigation guide from implementation to design rationale. Use this as your primary reference for understanding how code components relate to their design specifications.

## Quick Reference: Code → Design Mapping

```
src/cursus/
├── core/           → Core System & Orchestration
├── api/            → I/O System & User Interfaces
├── steps/          → Step Library System
├── step_catalog/   → Auto-Discovery System
├── registry/       → Component Registration
├── validation/     → Validation Framework
├── pipeline_catalog/ → Pipeline Sharing
├── cli/            → Command-Line Interface
├── mods/           → MODS Integration
└── workspace/      → Multi-Developer Support
```

---

## 1. Core System (`src/cursus/core/`)

The foundation of automatic pipeline generation through specification-driven design.

### 1.1 Pipeline Assembly (`core/assembler/`)

**Code Components:**
- `pipeline_assembler.py` - Message passing orchestration
- `pipeline_template_base.py` - Template foundation class

**Related Design Docs:**
- [Pipeline Assembler Design](../1_design/pipeline_assembler.md) - Message passing algorithm
- [Pipeline Template Base](../1_design/pipeline_template_base.md) - Template architecture
- [Specification-Driven Design](../1_design/specification_driven_design.md) - Core philosophy
- [Dynamic Template System](../1_design/dynamic_template_system.md) - Runtime generation

**Key Concepts:**
- Message passing algorithm for automatic step connection
- Template-based pipeline construction
- Declarative pipeline assembly

### 1.2 Pipeline Compilation (`core/compiler/`)

**Code Components:**
- `dag_compiler.py` - DAG to pipeline compilation
- `dynamic_template.py` - Runtime template generation
- `validation.py` - Compilation validation

**Related Design Docs:**
- [Pipeline Compiler Design](../1_design/pipeline_compiler.md) - Compilation architecture
- [DAG to Template](../1_design/dag_to_template.md) - Transformation process
- [Dynamic Template System](../1_design/dynamic_template_system.md) - Dynamic generation
- [MODS DAG Compiler](../1_design/mods_dag_compiler_design.md) - MODS integration

**Key Concepts:**
- DAG compilation to SageMaker pipelines
- Pipeline optimization
- Runtime template generation

### 1.3 Dependency Resolution (`core/deps/`)

**Code Components:**
- `dependency_resolver.py` - Automatic dependency matching
- `semantic_matcher.py` - Keyword-based matching
- `specification_registry.py` - Step metadata registry
- `property_reference.py` - SageMaker property handling

**Related Design Docs:**
- [Dependency Resolver](../1_design/dependency_resolver.md) - Resolution algorithm
- [Dependency Resolution System](../1_design/dependency_resolution_system.md) - System architecture
- [Enhanced Dependency Validation](../1_design/enhanced_dependency_validation_design.md) - Validation rules
- [Specification Registry](../1_design/specification_registry.md) - Registry design
- [Enhanced Property Reference](../1_design/enhanced_property_reference.md) - Property handling

**Key Concepts:**
- Semantic keyword matching
- Automatic input/output wiring
- Compatibility scoring

### 1.4 Configuration Management (`core/config_fields/`)

**Code Components:**
- Config field managers
- Configuration validation
- Field categorization

**Related Design Docs:**
- [Config Manager Three-Tier Implementation](../1_design/config_manager_three_tier_implementation.md)
- [Config Field Categorization](../1_design/config_field_categorization_consolidated.md)
- [Three-Tier Config Design](../1_design/config_tiered_design.md)
- [Config Resolution Enhancements](../1_design/config_resolution_enhancements.md)
- [Config Types Format](../1_design/config_types_format.md)

**Key Concepts:**
- Three-tier configuration architecture
- Field categorization and validation
- Configuration portability

### 1.5 Base Components (`core/base/`)

**Code Components:**
- Foundation classes
- Common utilities
- Base abstractions

**Related Design Docs:**
- [Design Principles](../0_developer_guide/design_principles.md)
- [Component Guide](../0_developer_guide/component_guide.md)

---

## 2. API & I/O System (`src/cursus/api/`)

User-facing interfaces and pipeline data structures.

### 2.1 DAG System (`api/dag/`)

**Code Components:**
- `base_dag.py` - Core DAG data structure
- `pipeline_dag_resolver.py` - DAG analysis and resolution

**Related Design Docs:**
- [Pipeline DAG](../1_design/pipeline_dag.md) - DAG architecture
- [Pipeline DAG Resolver](../1_design/pipeline_dag_resolver_design.md) - Resolution system
- [DAG to Template](../1_design/dag_to_template.md) - Transformation

**Key Concepts:**
- Directed acyclic graph representation
- Topological sorting
- Dependency tracking

### 2.2 Configuration Factory (`api/factory/`)

**Code Components:**
- `dag_config_factory.py` - Interactive configuration widgets

**Related Design Docs:**
- [DAG Config Factory Design](../1_design/dag_config_factory_design.md)
- [Generalized Config UI](../1_design/generalized_config_ui_design.md)
- [Nested Config UI](../1_design/nested_config_ui_design.md)
- [Essential Inputs Notebook](../1_design/essential_inputs_notebook_design_revised.md)

**Key Concepts:**
- Interactive Jupyter widgets
- User-friendly configuration
- Config serialization

---

## 3. Step Library System (`src/cursus/steps/`)

Comprehensive catalog of reusable ML pipeline components.

### 3.1 Step Specifications (`steps/specs/`)

**Code Components:**
- Step specification definitions
- Dependency declarations
- Output specifications

**Related Design Docs:**
- [Step Specification](../1_design/step_specification.md)
- [Specification-Driven Design](../1_design/specification_driven_design.md)
- [Step Contract](../1_design/step_contract.md)

**Key Concepts:**
- Declarative step metadata
- Dependency specifications
- Semantic keyword matching

### 3.2 Step Builders (`steps/builders/`)

**Code Components:**
- Step builder implementations
- SageMaker step creation
- Builder patterns

**Related Design Docs:**
- [Step Builder](../1_design/step_builder.md)
- [Step Builder Patterns Summary](../1_design/step_builder_patterns_summary.md)
- [Training Step Builder Patterns](../1_design/training_step_builder_patterns.md)
- [Processing Step Builder Patterns](../1_design/processing_step_builder_patterns.md)
- [Transform Step Builder Patterns](../1_design/transform_step_builder_patterns.md)
- [CreateModel Step Builder Patterns](../1_design/createmodel_step_builder_patterns.md)
- [Tuning Step Builder Patterns](../1_design/tuning_step_builder_patterns.md)
- [Conditional Step Builder Patterns](../1_design/conditional_step_builder_patterns.md)
- [Bedrock Processing Step Builder](../1_design/bedrock_processing_step_builder_patterns.md)
- [Label Ruleset Execution Step Patterns](../1_design/label_ruleset_execution_step_patterns.md)
- [Label Ruleset Generation Step Patterns](../1_design/label_ruleset_generation_step_patterns.md)

**Key Concepts:**
- Builder pattern implementation
- Step-specific creation logic
- Message-aware step creation

### 3.3 Step Configurations (`steps/configs/`)

**Code Components:**
- Configuration classes per step type
- Hyperparameter definitions
- Validation logic

**Related Design Docs:**
- [Three-Tier Config Design Implementation](../0_developer_guide/three_tier_config_design.md) - **Primary implementation guide**
- [Three-Tier Config Design](../1_design/config_tiered_design.md) - Architecture design
- [Config Portability Path Resolution](../1_design/config_portability_path_resolution_design.md) - Portable path implementation
- [Config Class Auto-Discovery](../1_design/config_class_auto_discovery_design.md)
- [Step Config Resolver](../1_design/step_config_resolver.md)
- [Hyperparameter Class Guide](../0_developer_guide/hyperparameter_class.md)

**Key Concepts:**
- Three-tier configuration architecture (Essential, System, Derived fields)
- Typed configuration classes with Pydantic v2
- Auto-discovery
- Default value management
- Portable path support for universal deployment

### 3.4 Script Contracts (`steps/contracts/`)

**Code Components:**
- Input/output contract definitions
- Environment variable specifications
- Contract validation

**Related Design Docs:**
- [Script Contract](../1_design/script_contract.md)
- [Environment Variable Contract Enforcement](../1_design/environment_variable_contract_enforcement.md)
- [Contract Discovery Manager](../1_design/contract_discovery_manager_design.md)

**Key Concepts:**
- Script interface definitions
- Contract alignment validation
- Environment variable contracts

### 3.5 Implementation Scripts (`steps/scripts/`)

**Code Components:**
- Actual step implementation scripts
- Training, preprocessing, evaluation code
- Feature engineering utilities

**Related Design Docs:**
- [Script Development Guide](../0_developer_guide/script_development_guide.md)
- [Script Testability Implementation](../0_developer_guide/script_testability_implementation.md)
- [Script Integration Testing System](../1_design/script_integration_testing_system_design.md)

**Key Concepts:**
- Script implementation patterns
- Testability
- Contract compliance

### 3.6 Hyperparameters (`steps/hyperparams/`)

**Code Components:**
- Hyperparameter definitions
- Default values
- Validation rules

**Related Design Docs:**
- [Hyperparameter Class Guide](../0_developer_guide/hyperparameter_class.md)
- [Default Values Provider](../1_design/default_values_provider_revised.md)
- [Smart Default Value Inheritance](../1_design/smart_default_value_inheritance_design.md)

**Key Concepts:**
- Hyperparameter management
- Default value inheritance
- Validation and constraints

---

## 4. Step Catalog (`src/cursus/step_catalog/`)

Automatic discovery and registration of step components.

**Code Components:**
- `step_catalog.py` - Main catalog system
- `spec_discovery.py` - Specification discovery
- `builder_discovery.py` - Builder discovery
- `config_discovery.py` - Configuration discovery
- `contract_discovery.py` - Contract discovery
- `script_discovery.py` - Script discovery
- `adapters/` - Integration adapters

**Related Design Docs:**
- [Unified Step Catalog System](../1_design/unified_step_catalog_system_design.md)
- [Step Catalog Component Architecture](../1_design/unified_step_catalog_component_architecture_design.md)
- [Step Catalog Expansion](../1_design/unified_step_catalog_system_expansion_design.md)
- [Step Catalog Search Space Management](../1_design/unified_step_catalog_system_search_space_management_design.md)
- [Step Catalog Integration Guide](../0_developer_guide/step_catalog_integration_guide.md)

**Key Concepts:**
- Auto-discovery mechanism
- Component registration
- Plugin architecture
- Metadata management

---

## 5. Registry (`src/cursus/registry/`)

Centralized registration and metadata management.

**Code Components:**
- `step_names.py` - Step type registry
- `hyperparameter_registry.py` - Hyperparameter registry
- `hybrid/` - Hybrid registry components

**Related Design Docs:**
- [Registry Single Source of Truth](../1_design/registry_single_source_of_truth.md)
- [Registry Manager](../1_design/registry_manager.md)
- [Config Registry](../1_design/config_registry.md)
- [Pipeline Registry](../1_design/pipeline_registry.md)
- [Registry-Based Step Name Generation](../1_design/registry_based_step_name_generation.md)
- [Hybrid Registry Standardization Enforcement](../1_design/hybrid_registry_standardization_enforcement_design.md)

**Key Concepts:**
- Centralized metadata
- Component registration
- Version management
- Hybrid registry pattern

---

## 6. Validation System (`src/cursus/validation/`)

Multi-level validation framework ensuring pipeline correctness.

**Code Components:**
- Alignment validation
- Builder validation
- Contract validation
- Script testing

**Related Design Docs:**
- [Validation Framework Guide](../0_developer_guide/validation_framework_guide.md)
- [Validation Checklist](../0_developer_guide/validation_checklist.md)
- [Alignment Rules](../0_developer_guide/alignment_rules.md)
- [Script Execution Spec](../1_design/script_execution_spec_design.md) - Script validation specification
- [Unified Alignment Tester Master Design](../1_design/unified_alignment_tester_master_design.md)
- [Unified Alignment Tester Architecture](../1_design/unified_alignment_tester_architecture.md)
- [SageMaker Step Type Aware Alignment Tester](../1_design/sagemaker_step_type_aware_unified_alignment_tester_design.md)
- [Universal Step Builder Test](../1_design/universal_step_builder_test.md)
- [Enhanced Universal Step Builder Tester](../1_design/enhanced_universal_step_builder_tester_design.md)
- [Step Builder Validation Rulesets](../1_design/step_builder_validation_rulesets_design.md)
- [Validation Configuration System Integration](../1_design/validation_configuration_system_integration_design.md)

**Validation Patterns by Step Type:**
- [Training Step Alignment Validation](../1_design/training_step_alignment_validation_patterns.md)
- [Processing Step Alignment Validation](../1_design/processing_step_alignment_validation_patterns.md)
- [Transform Step Alignment Validation](../1_design/transform_step_alignment_validation_patterns.md)
- [CreateModel Step Alignment Validation](../1_design/createmodel_step_alignment_validation_patterns.md)
- [RegisterModel Step Alignment Validation](../1_design/registermodel_step_alignment_validation_patterns.md)
- [Utility Step Alignment Validation](../1_design/utility_step_alignment_validation_patterns.md)

**Key Concepts:**
- Four-level validation hierarchy
- Alignment validation
- Builder testing
- Script validation

---

## 7. Pipeline Catalog (`src/cursus/pipeline_catalog/`)

Extended catalog for sharing complete pipeline structures.

**Code Components:**
- Pipeline catalog components
- Pipeline templates
- MODS integration

**Related Design Docs:**
- [Pipeline Catalog Design](../1_design/pipeline_catalog_design.md)
- [Pipeline Catalog Zettelkasten Refactoring](../1_design/pipeline_catalog_zettelkasten_refactoring.md)
- [Expanded Pipeline Catalog MODS Integration](../1_design/expanded_pipeline_catalog_mods_integration.md)
- [Pipeline Catalog Integration Guide](../0_developer_guide/pipeline_catalog_integration_guide.md)
- [Zettelkasten Pipeline Catalog Utilities](../1_design/zettelkasten_pipeline_catalog_utilities.md)

**Key Concepts:**
- Pipeline sharing
- Template management
- MODS integration
- Version control

---

## 8. Command-Line Interface (`src/cursus/cli/`)

Tools for pipeline compilation, validation, and deployment.

**Code Components:**
- CLI command implementations
- Pipeline compilation tools
- Validation commands

**Related Design Docs:**
- [CLI Pipeline Compilation Tools](../1_design/cli_pipeline_compilation_tools_design.md)
- [Workspace CLI Reference](../01_developer_guide_workspace_aware/ws_workspace_cli_reference.md)
- [Workspace-Aware CLI Design](../1_design/workspace_aware_cli_design.md)

**Key Concepts:**
- Command-line tools
- Pipeline compilation
- Validation commands
- Deployment utilities

---

## 9. MODS Integration (`src/cursus/mods/`)

MODS-specific extensions and integrations.

**Code Components:**
- MODS-specific step implementations
- MODS DAG compilation
- MODS pipeline templates
- Execution document generator

**Related Design Docs:**
- [MODS DAG Compiler](../1_design/mods_dag_compiler_design.md)
- [Expanded Pipeline Catalog MODS Integration](../1_design/expanded_pipeline_catalog_mods_integration.md)
- [Standalone Execution Document Generator Design](../1_design/standalone_execution_document_generator_design.md)

**Key Concepts:**
- MODS integration
- Custom DAG compilation
- MODS-specific features
- Standalone execution documentation

---

## 10. Workspace System (`src/cursus/workspace/`)

Multi-developer environment support and configuration isolation.

**Code Components:**
- Workspace management
- Configuration isolation
- Multi-user support

**Related Design Docs:**
- [Workspace-Aware System Master Design](../1_design/workspace_aware_system_master_design.md)
- [Workspace-Aware Core System](../1_design/workspace_aware_core_system_design.md)
- [Workspace-Aware Step Catalog Integration](../1_design/workspace_aware_system_step_catalog_integration_design.md)
- [Workspace Setup Guide](../01_developer_guide_workspace_aware/ws_workspace_setup_guide.md)
- [Workspace CLI Reference](../01_developer_guide_workspace_aware/ws_workspace_cli_reference.md)
- [Workspace Adding New Pipeline Step](../01_developer_guide_workspace_aware/ws_adding_new_pipeline_step.md)
- [Workspace Hybrid Registry Integration](../01_developer_guide_workspace_aware/ws_hybrid_registry_integration.md)

**Key Concepts:**
- Multi-developer support
- Configuration isolation
- Workspace-aware operations
- Collaborative development

---

## Cross-Cutting Concerns

### Configuration & Path Resolution

**Related Design Docs:**
- [Config Portability Path Resolution](../1_design/config_portability_path_resolution_design.md)
- [Deployment Context Agnostic Path Resolution](../1_design/deployment_context_agnostic_path_resolution_design.md)
- [Hybrid Strategy Deployment Path Resolution](../1_design/hybrid_strategy_deployment_path_resolution_design.md)
- [Flexible File Resolver](../1_design/flexible_file_resolver_design.md)

### Testing & Quality

**Related Design Docs:**
- [Pipeline Testing Spec](../1_design/pipeline_testing_spec_design.md)
- [Pipeline Testing Spec Builder](../1_design/pipeline_testing_spec_builder_design.md)
- [Universal Step Builder Test Scoring](../1_design/universal_step_builder_test_scoring.md)
- [Universal Step Builder Test Step Catalog Integration](../1_design/universal_step_builder_test_step_catalog_integration.md)

### Advanced Features

**Related Design Docs:**
- [Agentic Workflow Design](../1_design/agentic_workflow_design.md)
- [AWS Step Functions Extension](../1_design/aws_step_functions_extension_design.md)
- [Atomic Processing Architecture](../1_design/atomic_processing_architecture_design.md)
- [Automatic Documentation Generation](../1_design/automatic_documentation_generation_design.md)

### Output Management

**Related Design Docs:**
- [Cursus Framework Output Management](../1_design/cursus_framework_output_management.md)
- [Model Evaluation Path Handling](../1_design/model_evaluation_path_handling.md)
- [Pipeline Execution Temp Dir Integration](../1_design/pipeline_execution_temp_dir_integration.md)

---

## Developer Guides

### Getting Started
- [Developer Guide](../0_developer_guide/README.md) - Complete development guide
- [Prerequisites](../0_developer_guide/prerequisites.md) - Setup requirements
- [Best Practices](../0_developer_guide/best_practices.md) - Coding standards
- [Common Pitfalls](../0_developer_guide/common_pitfalls.md) - Avoid mistakes

### Component Development
- [Adding New Pipeline Step](../0_developer_guide/adding_new_pipeline_step.md) - Step creation workflow
- [Step Builder Guide](../0_developer_guide/step_builder.md) - Builder implementation
- [Script Development Guide](../0_developer_guide/script_development_guide.md) - Script implementation
- [Step Specification Guide](../0_developer_guide/step_specification.md) - Spec creation
- [Script Contract Guide](../0_developer_guide/script_contract.md) - Contract definition
- [Creation Process](../0_developer_guide/creation_process.md) - Step-by-step process

### Standards & Rules
- [Alignment Rules](../0_developer_guide/alignment_rules.md) - Component alignment
- [Standardization Rules](../0_developer_guide/standardization_rules.md) - Naming conventions
- [Design Principles](../0_developer_guide/design_principles.md) - Architectural guidelines
- [Component Guide](../0_developer_guide/component_guide.md) - Component overview

### Integration Guides
- [Step Catalog Integration](../0_developer_guide/step_catalog_integration_guide.md)
- [Pipeline Catalog Integration](../0_developer_guide/pipeline_catalog_integration_guide.md)
- [Config Field Manager Guide](../0_developer_guide/config_field_manager_guide.md)

---

## Navigation Guide

### By Development Task

**Creating a New Step:**
1. Start: [Adding New Pipeline Step](../0_developer_guide/adding_new_pipeline_step.md)
2. Specification: [Step Specification](../1_design/step_specification.md)
3. Builder: [Step Builder](../1_design/step_builder.md)
4. Script: [Script Development Guide](../0_developer_guide/script_development_guide.md)
5. Validation: [Validation Framework](../0_developer_guide/validation_framework_guide.md)

**Understanding Core System:**
1. Overview: [Cursus Package Overview](./cursus_package_overview.md)
2. Architecture: [Specification-Driven Design](../1_design/specification_driven_design.md)
3. Assembly: [Pipeline Assembler](../1_design/pipeline_assembler.md)
4. Dependencies: [Dependency Resolver](../1_design/dependency_resolver.md)

**Working with Configuration:**
1. Architecture: [Three-Tier Config Design](../1_design/config_tiered_design.md)
2. Management: [Config Manager Implementation](../1_design/config_manager_three_tier_implementation.md)
3. UI: [DAG Config Factory](../1_design/dag_config_factory_design.md)
4. Guide: [Config Field Manager Guide](../0_developer_guide/config_field_manager_guide.md)

**Validation & Testing:**
1. Framework: [Validation Framework Guide](../0_developer_guide/validation_framework_guide.md)
2. Alignment: [Unified Alignment Tester](../1_design/unified_alignment_tester_master_design.md)
3. Testing: [Universal Step Builder Test](../1_design/universal_step_builder_test.md)
4. Checklist: [Validation Checklist](../0_developer_guide/validation_checklist.md)

### By System Component

**Core System:**
- `core/assembler/` → [Pipeline Assembler](../1_design/pipeline_assembler.md)
- `core/compiler/` → [Pipeline Compiler](../1_design/pipeline_compiler.md)
- `core/deps/` → [Dependency Resolver](../1_design/dependency_resolver.md)
- `core/config_fields/` → [Config Manager](../1_design/config_manager_three_tier_implementation.md)

**Step System:**
- `steps/specs/` → [Step Specification](../1_design/step_specification.md)
- `steps/builders/` → [Step Builder](../1_design/step_builder.md)
- `steps/contracts/` → [Script Contract](../1_design/script_contract.md)
- `steps/scripts/` → [Script Development](../0_developer_guide/script_development_guide.md)

**API System:**
- `api/dag/` → [Pipeline DAG](../1_design/pipeline_dag.md)
- `api/factory/` → [DAG Config Factory](../1_design/dag_config_factory_design.md)

---

## Related Entry Points

- [Cursus Package Overview](./cursus_package_overview.md) - Executive summary and architecture
- [Core System & MODS Integration Index](./core_and_mods_systems_index.md) - Core orchestration and MODS integration
- [Step Design and Documentation Index](./step_design_and_documentation_index.md) - Step-specific patterns
- [Processing Steps Index](./processing_steps_index.md) - Complete processing step catalog
- [XGBoost Pipelines Index](./xgboost_pipelines_index.md) - Complete XGBoost pipeline variants catalog

---

## Maintenance Notes

**Last Updated:** 2025-11-09

**Update Triggers:**
- New code component added
- New design doc created
- Component relationships change
- Architecture refactoring

**Maintenance Guidelines:**
- Keep mappings current with code changes
- Add new design docs as created
- Update cross-references when docs move
- Verify links periodically

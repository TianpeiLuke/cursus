---
tags:
  - entry_point
  - documentation
  - core_system
  - mods_integration
  - overview
keywords:
  - core system
  - pipeline assembler
  - dag compiler
  - dynamic template
  - config manager
  - path resolution
  - mods integration
  - execution document generator
topics:
  - core system architecture
  - mods integration
  - pipeline orchestration
  - configuration management
language: python
date of note: 2025-11-09
---

# Core System & MODS Integration Index

## Overview

This index card serves as the comprehensive navigation hub for the Cursus Core System and MODS Integration documentation. The Core System provides the foundational orchestration, compilation, and configuration management capabilities, while MODS Integration extends these capabilities for MODS-specific workflows and execution document generation.

## Quick Navigation

```
Core & MODS Systems
├── Core System (src/cursus/core/)
│   ├── Pipeline Orchestration   → assembler/
│   ├── DAG Compilation          → compiler/
│   ├── Base Components          → base/
│   ├── Configuration Management → config_fields/
│   └── Path Resolution          → utils/
│
└── MODS Integration (src/cursus/mods/)
    └── Execution Documents      → exe_doc/
```

---

## 1. Core System (`src/cursus/core/`)

The foundation of automatic pipeline generation through specification-driven design and intelligent orchestration.

### 1.1 Pipeline Orchestration (`core/assembler/`)

**Code Components:**
- `pipeline_assembler.py` - Message passing orchestration algorithm
- `pipeline_template_base.py` - Abstract template foundation class

**Related Design Docs:**
- [Pipeline Assembler](../1_design/pipeline_assembler.md) - Message passing algorithm architecture
- [Pipeline Template Base](../1_design/pipeline_template_base.md) - Template foundation and patterns
- [Specification-Driven Design](../1_design/specification_driven_design.md) - Core design philosophy
- [Pipeline Execution Temp Dir Integration](../1_design/pipeline_execution_temp_dir_integration.md) - Parameter flow for temp directories

**Key Concepts:**
- Message passing algorithm for automatic step connection
- Template-based pipeline construction
- Declarative pipeline assembly
- Step builder orchestration

### 1.2 DAG Compilation (`core/compiler/`)

**Code Components:**
- `dag_compiler.py` - DAG to SageMaker Pipeline compilation
- `dynamic_template.py` - Runtime template generation
- `validation.py` - Compilation validation
- `exceptions.py` - Compiler-specific exceptions
- `name_generator.py` - Pipeline naming utilities

**Related Design Docs:**
- [Dynamic Template System](../1_design/dynamic_template_system.md) - **PRIMARY** - Dynamic template generation
- [Pipeline DAG](../1_design/pipeline_dag.md) - DAG structure and operations
- [DAG to Template](../1_design/dag_to_template.md) - Transformation process
- [CLI Pipeline Compilation Tools](../1_design/cli_pipeline_compilation_tools_design.md) - CLI compilation interface
- [Pipeline Execution Temp Dir Integration](../1_design/pipeline_execution_temp_dir_integration.md) - Parameter propagation architecture

**Key Concepts:**
- DAG compilation to SageMaker pipelines
- Pipeline optimization
- Runtime template generation
- Validation and error handling
- Parameter flow architecture

### 1.3 Base Components (`core/base/`)

**Code Components:**
- `builder_base.py` - Base step builder implementation
- `config_base.py` - Base configuration classes
- `contract_base.py` - Base contract definitions
- `enums.py` - System enumerations
- `hyperparameters_base.py` - Base hyperparameter classes
- `specification_base.py` - Base specification structures

**Related Design Docs:**
- [Step Builder](../1_design/step_builder.md) - Builder pattern architecture
- [Step Specification](../1_design/step_specification.md) - Specification format
- [Script Contract](../1_design/script_contract.md) - Contract definitions
- [Three-Tier Config Design](../1_design/config_tiered_design.md) - Configuration architecture
- [Hyperparameter Class Guide](../0_developer_guide/hyperparameter_class.md) - Hyperparameter patterns

**Key Concepts:**
- Foundation classes for all components
- Common abstractions and interfaces
- Extensible base patterns
- Type definitions

### 1.4 Configuration Management (`core/config_fields/`)

**Code Components:**
- `unified_config_manager.py` - Central configuration management
- `step_catalog_aware_categorizer.py` - Field categorization with catalog integration
- `type_aware_config_serializer.py` - Type-aware serialization
- `inheritance_aware_field_generator.py` - Field generation with inheritance
- `constants.py` - Configuration constants

**Related Design Docs:**
- [Config Manager Three-Tier Implementation](../1_design/config_manager_three_tier_implementation.md) - **PRIMARY** - Three-tier config system
- [Config Field Categorization](../1_design/config_field_categorization_consolidated.md) - Field classification
- [Config Tiered Design](../1_design/config_tiered_design.md) - Architecture principles
- [Config Resolution Enhancements](../1_design/config_resolution_enhancements.md) - Resolution improvements
- [Config Types Format](../1_design/config_types_format.md) - Type specifications
- [Step Config Resolver](../1_design/step_config_resolver.md) - DAG node to config mapping

**Key Concepts:**
- Three-tier configuration architecture (Essential, System, Derived)
- Field categorization and validation
- Type-aware serialization
- Inheritance-aware field generation
- Step catalog integration

### 1.5 Path Resolution Utilities (`core/utils/`)

**Code Components:**
- `hybrid_path_resolution.py` - Hybrid path resolution strategies
- `generic_path_discovery.py` - Generic path discovery utilities

**Related Design Docs:**
- [Hybrid Strategy Deployment Path Resolution](../1_design/hybrid_strategy_deployment_path_resolution_design.md) - **PRIMARY** - Hybrid resolution
- [Config Portability Path Resolution](../1_design/config_portability_path_resolution_design.md) - Portable paths
- [Deployment Context Agnostic Path Resolution](../1_design/deployment_context_agnostic_path_resolution_design.md) - Context-agnostic resolution
- [Cursus Package Portability Architecture](../1_design/cursus_package_portability_architecture_design.md) - Overall portability

**Key Concepts:**
- Hybrid path resolution (file-based + importlib)
- Deployment-agnostic path handling
- Universal compatibility (dev, PyPI, Docker, Lambda)
- Portable script paths

---

## 2. MODS Integration (`src/cursus/mods/`)

MODS-specific extensions for execution document generation and workflow integration.

### 2.1 Execution Document Generation (`mods/exe_doc/`)

**Code Components:**
- `generator.py` - Main execution document generator
- `base.py` - Base classes for document generation
- `cradle_helper.py` - Cradle service integration
- `registration_helper.py` - Model registration utilities
- `utils.py` - Document generation utilities

**Related Design Docs:**
- [Standalone Execution Document Generator Design](../1_design/standalone_execution_document_generator_design.md) - **PRIMARY** - Complete architecture
- [MODS DAG Compiler Design](../1_design/mods_dag_compiler_design.md) - MODS compilation
- [Expanded Pipeline Catalog MODS Integration](../1_design/expanded_pipeline_catalog_mods_integration.md) - Catalog integration
- [Pipeline Execution Temp Dir Integration](../1_design/pipeline_execution_temp_dir_integration.md) - Temp directory handling

**Analysis Documents:**
- [Execution Document Filling Analysis](../4_analysis/execution_document_filling_analysis.md) - Analysis that motivated the design

**Key Concepts:**
- Standalone execution document filling
- Complete independence from pipeline generation system
- Simplified input (PipelineDAG + config)
- MODS workflow integration
- Cradle and registration helpers

---

## 3. Cross-Cutting Concerns

### 3.1 Pipeline Templates & Orchestration

**Template System Hierarchy:**
1. **Pipeline Template Base** (`core/assembler/pipeline_template_base.py`) - Abstract foundation
2. **Dynamic Pipeline Template** (`core/compiler/dynamic_template.py`) - Runtime generation
3. **MODS Template Decorator** (MODS integration) - Enhanced metadata

**Related Design Docs:**
- [Pipeline Template Base](../1_design/pipeline_template_base.md) - Foundation patterns
- [Dynamic Template System](../1_design/dynamic_template_system.md) - Dynamic generation
- [Pipeline Template Builder V2](../1_design/pipeline_template_builder_v2.md) - Modern orchestration
- [Pipeline Template Builder V1](../1_design/pipeline_template_builder_v1.md) - Legacy system
- [Pipeline Execution Temp Dir Integration](../1_design/pipeline_execution_temp_dir_integration.md) - Parameter flow through template hierarchy

### 3.2 Configuration System Integration

**Configuration Flow:**
```
Config Files → Unified Config Manager → Field Categorization → 
Step Catalog Integration → Step Builders → Pipeline Assembly
```

**Related Design Docs:**
- [Config Manager Three-Tier Implementation](../1_design/config_manager_three_tier_implementation.md)
- [Config Class Auto-Discovery](../1_design/config_class_auto_discovery_design.md)
- [Workspace-Aware Config Manager](../1_design/workspace_aware_config_manager_design.md)

### 3.3 Workspace Integration

**Workspace-Aware Extensions:**
- Core System extensions for multi-developer support
- Configuration isolation and merging
- Step catalog workspace scoping

**Related Design Docs:**
- [Workspace-Aware System Master Design](../1_design/workspace_aware_system_master_design.md)
- [Workspace-Aware Core System](../1_design/workspace_aware_core_system_design.md)
- [Workspace-Aware CLI](../1_design/workspace_aware_cli_design.md)
- [Workspace-Aware Step Catalog Integration](../1_design/workspace_aware_system_step_catalog_integration_design.md)

---

## 4. Implementation Plans & Roadmaps

### 4.1 Path Resolution Implementation

- [Config Portability Path Resolution Implementation Plan](../2_project_planning/2025-09-20_config_portability_path_resolution_implementation_plan.md)
- [Hybrid Strategy Deployment Path Resolution Implementation Plan](../2_project_planning/2025-09-22_hybrid_strategy_deployment_path_resolution_implementation_plan.md)
- [MODS Lambda Sibling Directory Path Resolution Fix](../2_project_planning/2025-09-22_mods_lambda_sibling_directory_path_resolution_fix_completion.md)
- [Simplified Path File-Based Resolution Implementation Plan](../2_project_planning/2025-09-22_simplified_path_file_based_resolution_implementation_plan.md)

### 4.2 Configuration System Implementation

- [Unified Config Manager Implementation Plan](../2_project_planning/2025-10-01_unified_config_manager_implementation_plan.md)
- [Pipeline Execution Temp Dir Implementation Plan](../2_project_planning/2025-09-18_pipeline_execution_temp_dir_implementation_plan.md)

### 4.3 MODS Integration Implementation

- [Execution Document Generator Implementation](related to standalone_execution_document_generator_design.md)
- [MODS Pipeline Catalog Integration](related to expanded_pipeline_catalog_mods_integration.md)

---

## 5. Analysis & Optimization

### 5.1 System Analysis

- [Dynamic Pipeline Template Design Principles Compliance Analysis](../4_analysis/dynamic_pipeline_template_design_principles_compliance_analysis.md)
- [Runtime Script Discovery Redundancy Analysis](../4_analysis/2025-10-16_runtime_script_discovery_redundancy_analysis.md)
- [Config Field Management System Analysis](../4_analysis/config_field_management_system_analysis.md)

### 5.2 Path Resolution Analysis

- [ImportLib Usage Systemic Deployment Portability Analysis](../4_analysis/2025-09-19_importlib_usage_systemic_deployment_portability_analysis.md)
- [MODS Pipeline Path Resolution Error Analysis](../.internal/mods_pipeline_path_resolution_error_analysis.md)
- [MODS Deployment Process Reconstruction](../.internal/mods_deployment_process_reconstruction.md)

### 5.3 MODS Analysis

- [Execution Document Filling Analysis](../4_analysis/execution_document_filling_analysis.md) - Comprehensive analysis of execution document architecture

---

## 6. Developer Guides

### 6.1 Core System Development

- [Design Principles](../0_developer_guide/design_principles.md) - Fundamental design philosophy
- [Component Guide](../0_developer_guide/component_guide.md) - Component overview
- [Best Practices](../0_developer_guide/best_practices.md) - Coding standards

### 6.2 Configuration Development

- [Three-Tier Config Design](../0_developer_guide/three_tier_config_design.md) - **PRIMARY** implementation guide
- [Config Field Manager Guide](../0_developer_guide/config_field_manager_guide.md) - Field management
- [Hyperparameter Class Guide](../0_developer_guide/hyperparameter_class.md) - Hyperparameter patterns

### 6.3 Step Development

- [Adding New Pipeline Step](../0_developer_guide/adding_new_pipeline_step.md) - Step creation workflow
- [Step Builder Guide](../0_developer_guide/step_builder.md) - Builder implementation
- [Step Specification Guide](../0_developer_guide/step_specification.md) - Spec creation

---

## 7. Related Systems

### 7.1 Dependency Resolution

**Code Location:** `src/cursus/core/deps/` (if exists, otherwise referenced)

**Design Docs:**
- [Dependency Resolver](../1_design/dependency_resolver.md)
- [Dependency Resolution System](../1_design/dependency_resolution_system.md)
- [Enhanced Dependency Validation](../1_design/enhanced_dependency_validation_design.md)

### 7.2 Step Catalog

**Code Location:** `src/cursus/step_catalog/`

**Design Docs:**
- [Unified Step Catalog System](../1_design/unified_step_catalog_system_design.md)
- [Step Catalog Component Architecture](../1_design/unified_step_catalog_component_architecture_design.md)

### 7.3 Validation System

**Code Location:** `src/cursus/validation/`

**Design Docs:**
- [Unified Alignment Tester Master Design](../1_design/unified_alignment_tester_master_design.md)
- [Unified Alignment Tester Architecture](../1_design/unified_alignment_tester_architecture.md)

---

## 8. Architecture Patterns

### 8.1 Message Passing Pattern

The core orchestration uses a message passing algorithm where:
1. Pipeline assembler coordinates step builders
2. Step builders declare their outputs via specifications
3. Dependency resolver matches outputs to inputs
4. Steps are connected automatically

**Reference:** [Pipeline Assembler](../1_design/pipeline_assembler.md)

### 8.2 Template Pattern

Templates provide structure for pipeline generation:
1. Abstract base defines interface
2. Dynamic templates handle runtime generation
3. MODS decorators add enhanced metadata
4. Configuration drives template instantiation

**Reference:** [Pipeline Template Base](../1_design/pipeline_template_base.md)

### 8.3 Three-Tier Configuration Pattern

Configuration is organized in three tiers:
1. **Essential Fields** - User-provided values
2. **System Fields** - Framework-managed values
3. **Derived Fields** - Computed from Essential/System

**Reference:** [Three-Tier Config Design](../0_developer_guide/three_tier_config_design.md)

---

## 9. Integration Points

### 9.1 With Step Library

**Integration:** Core system provides builders, configs, and specs that step library extends

**Flow:** Base Components → Step Specifications → Step Builders → Step Configs → Pipeline Assembly

### 9.2 With API Layer

**Integration:** Core compiler and assembler used by API's DAG factory

**Flow:** API DAG → DAG Compiler → Dynamic Template → Pipeline Assembler → SageMaker Pipeline

### 9.3 With MODS Workflow

**Integration:** MODS extends core compilation with execution document generation

**Flow:** Pipeline DAG → MODS Compiler → Execution Document Generator → MODS Workflow System

---

## 10. Quick Reference

### Common Development Tasks

**Creating a New Template:**
1. Extend `PipelineTemplateBase` → [Template Base](../1_design/pipeline_template_base.md)
2. Implement abstract methods → [Dynamic Template System](../1_design/dynamic_template_system.md)
3. Register with compiler → [DAG Compiler](../1_design/cli_pipeline_compilation_tools_design.md)

**Adding Configuration Fields:**
1. Define in config class → [Three-Tier Config](../0_developer_guide/three_tier_config_design.md)
2. Categorize fields → [Field Categorization](../1_design/config_field_categorization_consolidated.md)
3. Test serialization → [Type-Aware Serializer](code reference)

**Implementing Path Resolution:**
1. Use hybrid strategy → [Hybrid Path Resolution](../1_design/hybrid_strategy_deployment_path_resolution_design.md)
2. Add portable paths → [Config Portability](../1_design/config_portability_path_resolution_design.md)
3. Test all deployment contexts → [Deployment Agnostic](../1_design/deployment_context_agnostic_path_resolution_design.md)

**Generating Execution Documents:**
1. Create DAG → [Pipeline DAG](../1_design/pipeline_dag.md)
2. Use standalone generator → [Execution Document Generator](../1_design/standalone_execution_document_generator_design.md)
3. Integrate with MODS → [MODS Integration](../1_design/expanded_pipeline_catalog_mods_integration.md)

---

## 11. System Statistics

### Core System Components

**Assembler:**
- `pipeline_assembler.py` - ~500+ lines (message passing orchestration)
- `pipeline_template_base.py` - ~300+ lines (abstract base class)

**Compiler:**
- `dag_compiler.py` - ~600+ lines (compilation engine)
- `dynamic_template.py` - ~400+ lines (runtime generation)
- `validation.py` - ~200+ lines (validation logic)

**Configuration:**
- `unified_config_manager.py` - ~400+ lines (central management)
- `step_catalog_aware_categorizer.py` - ~300+ lines (categorization)
- `type_aware_config_serializer.py` - ~250+ lines (serialization)

### MODS Integration

**Execution Documents:**
- `generator.py` - ~300+ lines (document generation)
- `cradle_helper.py` - ~150+ lines (Cradle integration)
- `registration_helper.py` - ~150+ lines (registration utilities)

---

## 12. Related Entry Points

- [Cursus Package Overview](./cursus_package_overview.md) - Executive summary and architecture
- [Cursus Code Structure Index](./cursus_code_structure_index.md) - Complete code-to-doc mapping
- [Step Design and Documentation Index](./step_design_and_documentation_index.md) - Step-specific patterns
- [Processing Steps Index](./processing_steps_index.md) - Processing step catalog

---

## Maintenance Notes

**Last Updated:** 2025-11-09

**Update Triggers:**
- New core component added
- New MODS feature implemented
- Configuration system changes
- Path resolution updates
- Template system modifications

**Maintenance Guidelines:**
- Keep design doc links current
- Update code component lists
- Track implementation plans
- Maintain integration point documentation
- Update statistics periodically

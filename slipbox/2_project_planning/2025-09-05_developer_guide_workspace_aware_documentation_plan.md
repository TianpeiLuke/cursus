---
tags:
  - project
  - planning
  - documentation
  - workspace_aware
  - developer_guide
keywords:
  - developer guide update
  - workspace awareness
  - hybrid registry system
  - documentation modernization
  - multi-workspace development
  - isolated project development
  - registry integration
  - step creation workflow
topics:
  - developer documentation
  - workspace management
  - registry system integration
  - development workflow
language: python
date of note: 2025-09-05
---

# Developer Guide Workspace-Aware Documentation Plan

## Executive Summary

This plan outlines the comprehensive update strategy for developer documentation to support the fully implemented hybrid registry system and workspace-aware development workflows. The plan addresses two distinct development approaches:

1. **Main Workspace Development**: Developers working directly in `src/cursus/` (traditional approach)
2. **Isolated Project Development**: Developers working in `development/projects/*/src/cursus_dev/` with hybrid registry access

## Background Context

### Current State Analysis

The hybrid registry system implementation is complete (Phases 0-5) with significant achievements:
- **54% code reduction** (2,837 → 1,285 lines)
- **UnifiedRegistryManager** providing drop-in registry replacement
- **Workspace-aware step resolution** with simple workspace priority
- **CLI workspace management** commands (`init-workspace`, `list-steps --workspace`)
- **Consolidated workspace system** with 26 components and 5-level validation

### Documentation Gap Analysis

Current developer guides in `slipbox/0_developer_guide/` are outdated and lack:
- Workspace awareness concepts
- Hybrid registry integration patterns
- Multi-project development workflows
- Modern CLI tooling usage
- Isolated development environment setup

## Plan Overview

### Phase 1: Update Existing Developer Guide (slipbox/0_developer_guide)
**Target Audience**: Developers working directly in main workspace (`src/cursus/`)
**Timeline**: 2 weeks
**Scope**: Modernize existing documentation for hybrid registry system

### Phase 2: Create Workspace-Aware Developer Guide (slipbox/01_workspace_aware_developer_guide)
**Target Audience**: Developers working in isolated projects (`development/projects/*/`)
**Timeline**: 3 weeks
**Scope**: New comprehensive guide for multi-project development

## Phase 1: Main Workspace Developer Guide Updates

### 1.1 Core Documents to Update

#### High Priority Updates

1. **`adding_new_pipeline_step.md`**
   - **Current Gap**: No workspace awareness, outdated registry patterns
   - **Updates Needed**:
     - Add UnifiedRegistryManager usage patterns
     - Include workspace context management
     - Update step registration examples with hybrid backend
     - Add CLI validation commands (`cursus validate-step`)
     - Include workspace-aware testing patterns

2. **`step_builder_registry_guide.md`**
   - **Current Gap**: References old registry classes, missing hybrid patterns
   - **Updates Needed**:
     - Replace old registry references with UnifiedRegistryManager
     - Add workspace context examples
     - Update @register_builder decorator usage
     - Include registry resolution strategies
     - Add troubleshooting for registry conflicts

3. **`step_builder_registry_usage.md`**
   - **Current Gap**: Outdated usage patterns
   - **Updates Needed**:
     - Modernize registry lookup examples
     - Add workspace-aware resolution examples
     - Include caching behavior documentation
     - Update performance considerations

#### Medium Priority Updates

4. **`creation_process.md`**
   - Add workspace validation steps
   - Include hybrid registry integration checkpoints
   - Update testing workflow with workspace awareness

5. **`validation_checklist.md`**
   - Add workspace context validation items
   - Include registry resolution verification
   - Update testing checklist for hybrid system

6. **`best_practices.md`**
   - Add workspace management best practices
   - Include registry performance optimization
   - Update code organization guidelines

### 1.2 New Sections to Add

#### Registry Integration Patterns

## Working with UnifiedRegistryManager

### Basic Usage
```python
from cursus.registry.hybrid.manager import UnifiedRegistryManager

# Get registry instance
registry = UnifiedRegistryManager()

# Set workspace context for development
registry.set_workspace_context("main")

# Register a step builder
registry.register_step_builder("my_step", MyStepBuilder)
```

#### Workspace Context Management
```markdown
## Workspace Context in Main Development

### Setting Context
- Default context: "main" workspace
- Explicit context setting for testing
- Context inheritance patterns

### Testing with Workspace Awareness
- Isolated test environments
- Context switching in tests
- Registry state management
```

### 1.3 Updated Workflow Examples

#### Modern Step Creation Workflow
1. **Design Phase**: Use workspace-aware design patterns
2. **Implementation**: Leverage UnifiedRegistryManager
3. **Registration**: Use hybrid registry backend
4. **Testing**: Include workspace context validation
5. **Integration**: Verify cross-workspace compatibility

## Phase 2: Workspace-Aware Developer Guide Creation

### 2.1 New Guide Structure (`slipbox/01_workspace_aware_developer_guide/`)

#### Self-Contained Core Documents (Symmetrical to slipbox/0_developer_guide)

1. **`ws_adding_new_pipeline_step.md`** - **Primary entry point** for workspace-aware development (adapted from main guide with workspace focus)
2. **`ws_README.md`** - Guide overview and navigation
3. **`ws_workspace_setup_guide.md`** - Initial project setup and initialization
4. **`ws_creation_process.md`** - Complete step creation workflow in `project/src/cursus_dev/` (adapted from main guide)
5. **`ws_step_builder.md`** - Step builder patterns for isolated projects (adapted from main guide)
6. **`ws_step_builder_registry_guide.md`** - Registry usage in workspace context (adapted from main guide)
7. **`ws_step_builder_registry_usage.md`** - Registry lookup examples for projects (adapted from main guide)
8. **`ws_component_guide.md`** - Component architecture in isolated projects (adapted from main guide)
9. **`ws_design_principles.md`** - Design principles for workspace development (adapted from main guide)
10. **`ws_best_practices.md`** - Best practices for isolated development (adapted from main guide)
11. **`ws_validation_checklist.md`** - Quality assurance for workspace projects (adapted from main guide)
12. **`ws_standardization_rules.md`** - Code standards for isolated projects (adapted from main guide)
13. **`ws_script_contract.md`** - Script interfaces in workspace context (adapted from main guide)
14. **`ws_step_specification.md`** - Step definitions for isolated projects (adapted from main guide)

#### Workspace-Specific Documents

15. **`ws_hybrid_registry_integration.md`** - Registry system usage patterns
16. **`ws_shared_code_access_patterns.md`** - Accessing `src/cursus` components
17. **`ws_workspace_cli_reference.md`** - CLI tools and commands
18. **`ws_testing_in_isolated_projects.md`** - Testing strategies
19. **`ws_deployment_and_integration.md`** - Moving code to production

#### Advanced Topics

20. **`ws_multi_project_collaboration.md`** - Working across projects
21. **`ws_workspace_migration_guide.md`** - Moving between workspaces
22. **`ws_troubleshooting_workspace_issues.md`** - Common problems and solutions
23. **`ws_performance_optimization.md`** - Workspace-specific optimizations

### 2.2 Key Content Areas

#### Workspace Setup and Initialization

## Project Initialization

### Creating a New Isolated Project
```bash
# Initialize new project workspace
cursus init-workspace --project project_delta --type isolated

# Verify workspace setup
cursus list-workspaces
cursus list-steps --workspace project_delta
```

### Project Structure
```
development/projects/project_delta/
├── src/cursus_dev/           # Custom project code (isolated from main src/cursus)
│   ├── steps/               # Project-specific step implementations
│   │   ├── __init__.py
│   │   ├── my_custom_step.py
│   │   └── builders/        # Step builders for project-specific steps
│   ├── configs/             # Project-specific configurations
│   └── utils/               # Project utility functions
├── tests/                   # Project-specific tests
│   ├── test_custom_steps.py
│   └── integration/
├── data/                    # Project data files
└── workspace_config.yaml    # Workspace configuration
```

#### Isolated Step Creation Process

## Creating Steps in Isolated Projects

### Step Creation Workflow (project/src/cursus_dev/steps)
```python
# File: development/projects/project_delta/src/cursus_dev/steps/my_custom_step.py

from cursus.core.step import Step
from cursus.registry.hybrid.manager import UnifiedRegistryManager

class MyCustomStep(Step):
    """Custom step implementation in isolated project environment."""
    
    def __init__(self, config):
        super().__init__(config)
        self.project_specific_param = config.get('project_param')
    
    def execute(self, input_data):
        # Project-specific step logic
        processed_data = self._custom_processing(input_data)
        return processed_data
    
    def _custom_processing(self, data):
        # Custom logic specific to this project
        return data

# Register step in project workspace
registry = UnifiedRegistryManager()
registry.set_workspace_context("project_delta")
registry.register_step("my_custom_step", MyCustomStep)
```

### Step Builder Creation (project/src/cursus_dev/steps/builders)
```python
# File: development/projects/project_delta/src/cursus_dev/steps/builders/my_custom_step_builder.py

from cursus.core.step_builder import StepBuilder
from ..my_custom_step import MyCustomStep

class MyCustomStepBuilder(StepBuilder):
    """Builder for project-specific custom step."""
    
    def build_step(self, config):
        return MyCustomStep(config)
    
    def validate_config(self, config):
        # Project-specific validation logic
        required_fields = ['project_param', 'input_path']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        return True

# Register builder in project workspace
registry = UnifiedRegistryManager()
registry.set_workspace_context("project_delta")
registry.register_step_builder("my_custom_step", MyCustomStepBuilder)
```

#### Hybrid Registry Usage Patterns
```markdown
## Registry Access Patterns

### Accessing Shared Components
```python
# Access shared steps from src/cursus
registry = UnifiedRegistryManager()
registry.set_workspace_context("project_delta")

# Shared step (from src/cursus)
shared_step = registry.get_step_builder("preprocessing_step")

# Project-specific step (from src/cursus_dev)
custom_step = registry.get_step_builder("project_delta_custom_step")
```

### Resolution Priority
1. Project-specific code (`src/cursus_dev`)
2. Shared code (`src/cursus`)
3. Fallback to default implementations
```

#### Development Workflow Integration
```markdown
## Daily Development Workflow

### 1. Workspace Activation
```bash
cd development/projects/project_delta
cursus activate-workspace project_delta
```

### 2. Development Cycle
- Edit code in `src/cursus_dev/`
- Access shared utilities from `src/cursus/`
- Test with workspace-aware registry
- Validate integration with shared components

### 3. Testing and Validation
```bash
# Run project-specific tests
cursus test --workspace project_delta

# Validate registry integration
cursus validate-registry --workspace project_delta

# Check for conflicts with shared code
cursus check-conflicts --workspace project_delta
```

## Implementation Strategy

### Phase 1 Implementation (Weeks 1-2)

#### Week 1: Core Document Updates
- Update `adding_new_pipeline_step.md` with hybrid registry patterns
- Modernize `step_builder_registry_guide.md` with UnifiedRegistryManager
- Revise `step_builder_registry_usage.md` with workspace examples

#### Week 2: Supporting Document Updates
- Update `creation_process.md` with workspace validation
- Revise `validation_checklist.md` for hybrid system
- Enhance `best_practices.md` with workspace guidelines

### Phase 2 Implementation (Weeks 3-5)

#### Week 3: Foundation and Core Symmetrical Documents
- Create `ws_adding_new_pipeline_step.md` (primary entry point, adapted from main guide with workspace focus)
- Create `ws_README.md` and `ws_workspace_setup_guide.md`
- Develop `ws_creation_process.md` (adapted from main guide for isolated projects)
- Build `ws_step_builder.md` (adapted from main guide for workspace context)
- Create `ws_step_builder_registry_guide.md` (adapted from main guide for workspace registry usage)

#### Week 4: Symmetrical Support Documents
- Create `ws_step_builder_registry_usage.md` (adapted from main guide for project-specific usage)
- Develop `ws_component_guide.md` (adapted from main guide for isolated project architecture)
- Build `ws_design_principles.md` (adapted from main guide for workspace development)
- Create `ws_best_practices.md` (adapted from main guide for isolated development)
- Develop `ws_validation_checklist.md` (adapted from main guide for workspace projects)

#### Week 5: Remaining Symmetrical and Workspace-Specific Documents
- Create `ws_standardization_rules.md` (adapted from main guide for isolated projects)
- Develop `ws_script_contract.md` (adapted from main guide for workspace context)
- Build `ws_step_specification.md` (adapted from main guide for isolated projects)
- Create `ws_hybrid_registry_integration.md`, `ws_shared_code_access_patterns.md`
- Develop `ws_workspace_cli_reference.md`, `ws_testing_in_isolated_projects.md`
- Build `ws_deployment_and_integration.md` and advanced topic documents
- Comprehensive testing and validation

## Key Differences Between Development Approaches

### Main Workspace Development (`src/cursus/`)
- **Direct code modification** in production codebase
- **Immediate integration** with existing components
- **Shared responsibility** for code quality and compatibility
- **Traditional testing** patterns with full system access
- **Registry context**: Default "main" workspace

### Isolated Project Development (`development/projects/*/`)
- **Isolated development environment** with project-specific code
- **Hybrid access** to both shared (`src/cursus`) and custom (`src/cursus_dev`) code
- **Project-specific responsibility** with controlled integration points
- **Workspace-aware testing** with registry context management
- **Registry context**: Project-specific workspace with fallback to shared

## Success Metrics

### Phase 1 Success Criteria
- [ ] All existing developer guide documents updated with hybrid registry patterns
- [ ] Workspace awareness integrated into step creation workflow
- [ ] Modern CLI tooling documented and examples provided
- [ ] Backward compatibility maintained for existing developers

### Phase 2 Success Criteria
- [ ] Complete workspace-aware developer guide created
- [ ] Isolated project development workflow fully documented
- [ ] Hybrid registry integration patterns clearly explained
- [ ] CLI reference documentation comprehensive and accurate
- [ ] Testing strategies for isolated development documented

### Overall Success Metrics
- **Developer Onboarding Time**: Reduce new developer onboarding from 2 weeks to 1 week
- **Documentation Coverage**: 100% coverage of hybrid registry features
- **Developer Satisfaction**: >90% satisfaction with updated documentation
- **Error Reduction**: 50% reduction in registry-related development errors

## Risk Mitigation

### Documentation Consistency Risk
- **Risk**: Inconsistency between main and workspace-aware guides
- **Mitigation**: Cross-reference validation and shared examples

### Developer Confusion Risk
- **Risk**: Confusion about which development approach to use
- **Mitigation**: Clear decision matrix and use case examples

### Maintenance Overhead Risk
- **Risk**: Increased documentation maintenance burden
- **Mitigation**: Automated validation tools and regular review cycles

## Conclusion

This comprehensive plan addresses the critical need to modernize developer documentation for the fully implemented hybrid registry system. By creating two complementary documentation sets, we support both traditional main workspace development and modern isolated project development workflows.

The phased approach ensures minimal disruption to current developers while providing a clear path to leverage the new workspace-aware capabilities. Success will be measured through improved developer experience, reduced onboarding time, and increased adoption of the hybrid registry system's advanced features.

## Related References

### Current Developer Guide Documents

#### Documents Requiring Updates
- [adding_new_pipeline_step.md](../0_developer_guide/adding_new_pipeline_step.md) - Core step creation process (needs hybrid registry integration)
- [step_builder_registry_guide.md](../0_developer_guide/step_builder_registry_guide.md) - Registry usage patterns (needs UnifiedRegistryManager updates)
- [step_builder_registry_usage.md](../0_developer_guide/step_builder_registry_usage.md) - Registry lookup examples (needs workspace awareness)
- [creation_process.md](../0_developer_guide/creation_process.md) - Development workflow (needs workspace validation)
- [validation_checklist.md](../0_developer_guide/validation_checklist.md) - Quality assurance (needs hybrid system checks)
- [best_practices.md](../0_developer_guide/best_practices.md) - Development guidelines (needs workspace management practices)

#### Supporting Documents
- [component_guide.md](../0_developer_guide/component_guide.md) - Component architecture overview
- [design_principles.md](../0_developer_guide/design_principles.md) - Core design philosophy
- [standardization_rules.md](../0_developer_guide/standardization_rules.md) - Code standardization requirements
- [script_contract.md](../0_developer_guide/script_contract.md) - Script interface specifications
- [step_specification.md](../0_developer_guide/step_specification.md) - Step definition standards

### Design Documents

#### Hybrid Registry System Design
- [hybrid_registry_standardization_enforcement_design.md](../1_design/hybrid_registry_standardization_enforcement_design.md) - Comprehensive standardization enforcement design with automated validation
- [workspace_aware_distributed_registry_design.md](../1_design/workspace_aware_distributed_registry_design.md) - Simplified hybrid registry architecture with UnifiedRegistryManager
- [config_registry.md](../1_design/config_registry.md) - Configuration registry patterns and integration
- [pipeline_registry.md](../1_design/pipeline_registry.md) - Pipeline component registry design

#### Workspace-Aware System Design
- [workspace_aware_system_master_design.md](../1_design/workspace_aware_system_master_design.md) - Master design for workspace-aware system architecture
- [workspace_aware_multi_developer_management_design.md](../1_design/workspace_aware_multi_developer_management_design.md) - Multi-developer workspace management system
- [workspace_aware_core_system_design.md](../1_design/workspace_aware_core_system_design.md) - Core workspace system design and implementation
- [global_vs_local_objects.md](../1_design/global_vs_local_objects.md) - Object scope and workspace isolation patterns
- [dependency_resolution_system.md](../1_design/dependency_resolution_system.md) - Cross-workspace dependency management
- [enhanced_dependency_validation_design.md](../1_design/enhanced_dependency_validation_design.md) - Validation framework for workspace dependencies

#### Configuration and Validation Design
- [config_driven_design.md](../1_design/config_driven_design.md) - Configuration-driven architecture principles
- [config_manager_three_tier_implementation.md](../1_design/config_manager_three_tier_implementation.md) - Three-tier configuration management
- [adaptive_configuration_management_system_revised.md](../1_design/adaptive_configuration_management_system_revised.md) - Dynamic configuration adaptation
- [enhanced_universal_step_builder_tester_design.md](../1_design/enhanced_universal_step_builder_tester_design.md) - Universal testing framework design

#### Integration and Workflow Design
- [adaptive_fluent_proxy_integration.md](../1_design/adaptive_fluent_proxy_integration.md) - Fluent API integration patterns
- [adaptive_specification_integration.md](../1_design/adaptive_specification_integration.md) - Specification system integration
- [pipeline_runtime_core_engine_design.md](../1_design/pipeline_runtime_core_engine_design.md) - Runtime engine architecture
- [pipeline_runtime_execution_layer_design.md](../1_design/pipeline_runtime_execution_layer_design.md) - Execution layer design

### Project Planning Documents

#### Hybrid Registry Migration and Implementation
- [2025-09-02_workspace_aware_hybrid_registry_migration_plan.md](2025-09-02_workspace_aware_hybrid_registry_migration_plan.md) - Complete migration plan (Phases 0-5) with 54% code reduction
- [2025-09-04_hybrid_registry_redundancy_reduction_plan.md](2025-09-04_hybrid_registry_redundancy_reduction_plan.md) - Redundancy reduction achieving 25% code reduction
- [2025-09-05_hybrid_registry_standardization_enforcement_implementation_plan.md](2025-09-05_hybrid_registry_standardization_enforcement_implementation_plan.md) - Standardization enforcement implementation

### Implementation Code References

#### Hybrid Registry Implementation
- [src/cursus/registry/hybrid/manager.py](../../src/cursus/registry/hybrid/manager.py) - UnifiedRegistryManager implementation
- [src/cursus/registry/hybrid/models.py](../../src/cursus/registry/hybrid/models.py) - Pydantic V2 models with enum validation
- [src/cursus/workspace/](../../src/cursus/workspace/) - Consolidated workspace system (26 components)
- [development/projects/](../../development/projects/) - Project isolation structure
- [development/README.md](../../development/README.md) - Multi-project development documentation

## Next Steps

1. **Stakeholder Review**: Present plan to development team for feedback
2. **Resource Allocation**: Assign technical writers and developers for implementation
3. **Timeline Confirmation**: Validate timeline with project priorities
4. **Implementation Kickoff**: Begin Phase 1 implementation with core document updates

---

*This plan document serves as the master reference for the developer guide modernization effort and should be updated as implementation progresses.*

---
tags:
  - llm_developer
  - migration_guide
  - architecture_modernization
  - prompt_templates
keywords:
  - 4-layer to 6-layer migration
  - workspace-aware development
  - UnifiedRegistryManager
  - three-tier configuration
  - prompt template updates
topics:
  - architecture migration
  - development workflow changes
  - template modernization
  - system evolution
language: markdown
date of note: 2025-09-05
---

# LLM Developer Prompt Templates Migration Guide

## Overview

This migration guide helps developers transition from the legacy 4-layer architecture to the modernized 6-layer architecture with workspace-aware development support. All prompt templates have been updated to reflect these architectural changes and provide comprehensive knowledge base references.

## Migration Summary

### Architecture Evolution: 4-Layer → 6-Layer

#### Legacy 4-Layer Architecture (Deprecated)
1. **Step Specifications**: Define inputs and outputs with logical names
2. **Script Contracts**: Define container paths for script inputs/outputs  
3. **Step Builders**: Connect specifications and contracts via SageMaker
4. **Processing Scripts**: Implement the actual business logic

#### Modern 6-Layer Architecture (Current)
1. **Step Specifications**: Define inputs and outputs with logical names and dependency relationships
2. **Script Contracts**: Define container paths and environment variables for script execution
3. **Processing Scripts**: Implement business logic using unified main function interface for testability
4. **Step Builders**: Connect specifications and contracts via SageMaker with UnifiedRegistryManager integration
5. **Configuration Classes**: Manage step parameters using three-tier field classification (Essential/System/Derived)
6. **Hyperparameters**: Handle ML-specific parameter tuning and optimization

### Key Modernization Features Added

#### 1. Workspace-Aware Development
- **Shared Workspace Development**: Direct access to `src/cursus/steps/` for experienced developers
- **Isolated Workspace Development**: Work in `development/projects/project_xxx/` with read-only access to shared components
- **Multi-Developer Collaboration**: Framework for team development with workspace isolation

#### 2. UnifiedRegistryManager System
- **Consolidated Registry**: Single registry system replacing legacy patterns
- **Workspace Context Handling**: Registry aware of workspace boundaries and contexts
- **Enhanced Step Discovery**: Improved step registration and discovery mechanisms

#### 3. Three-Tier Configuration Design
- **Essential Fields**: Core business logic parameters
- **System Fields**: Infrastructure and execution parameters
- **Derived Fields**: Computed or inferred parameters
- **Better Maintainability**: Clear categorization improves code organization

#### 4. Enhanced Validation Framework
- **Two-Level Validation**: Combines LLM analysis with strict tool validation
- **Workspace-Aware Validation**: Validation respects workspace boundaries
- **Comprehensive Coverage**: Alignment, standardization, and integration validation

#### 5. Script Testability Implementation
- **Unified Main Function Interface**: Parameterized main functions for better testability
- **Environment Collection Patterns**: Standardized environment variable handling
- **Hybrid Execution Support**: Support for both container and local execution modes

## Template-by-Template Migration Guide

### Core Workflow Templates

#### 1. step1_initial_planner_prompt_template.md
**Changes Made**:
- ✅ Updated to 6-layer architecture description
- ✅ Added workspace-aware development context
- ✅ Added comprehensive knowledge base references
- ✅ Updated registry references to `step_names_original.py`
- ✅ Added three-tier configuration design references
- ✅ Added script testability implementation guidance

**Migration Impact**: 
- Agents now understand modern architecture layers
- Planning includes workspace context considerations
- Enhanced knowledge base provides better domain understanding

#### 2. step2_plan_validator_prompt_template.md  
**Changes Made**:
- ✅ Updated architecture context to 6-layer design
- ✅ Added workspace-aware validation considerations
- ✅ Enhanced knowledge base with modern references
- ✅ Updated registry integration patterns
- ✅ Added three-tier configuration validation

**Migration Impact**:
- Plan validation now covers modern architectural patterns
- Workspace-aware validation ensures proper isolation
- Enhanced validation criteria improve plan quality

#### 3. step3_revision_planner_prompt_template.md
**Changes Made**:
- ✅ Updated design principles to include 6-layer architecture
- ✅ Added workspace-aware development references
- ✅ Enhanced validation framework references
- ✅ Updated registry integration patterns
- ✅ Added script testability implementation guidance

**Migration Impact**:
- Plan revisions consider modern architectural patterns
- Workspace context influences revision strategies
- Enhanced knowledge base improves revision quality

#### 4. step4_programmer_prompt_template.md
**Changes Made**:
- ✅ Updated to 6-layer architecture understanding
- ✅ Added comprehensive workspace-aware references
- ✅ Enhanced implementation examples and patterns
- ✅ Updated registry integration to UnifiedRegistryManager
- ✅ Added three-tier configuration implementation guidance

**Migration Impact**:
- Code generation follows modern architectural patterns
- Workspace-aware implementation ensures proper isolation
- Enhanced examples improve code quality

#### 5. step5a_two_level_validation_agent_prompt_template.md
**Changes Made**:
- ✅ Updated system architecture to 6-layer design
- ✅ Added workspace-aware validation tools
- ✅ Enhanced knowledge base with comprehensive references
- ✅ Updated registry validation patterns
- ✅ Added three-tier configuration validation

**Migration Impact**:
- Validation covers modern architectural patterns
- Workspace-aware validation tools ensure proper isolation
- Enhanced validation provides better coverage

#### 6. step5b_two_level_standardization_validation_agent_prompt_template.md
**Changes Made**:
- ✅ Updated design principles to 6-layer architecture
- ✅ Added workspace-aware development references
- ✅ Enhanced standardization validation patterns
- ✅ Updated registry integration patterns
- ✅ Added three-tier configuration standardization

**Migration Impact**:
- Standardization validation covers modern patterns
- Workspace-aware standardization ensures consistency
- Enhanced validation improves code quality

#### 7. step6_code_refinement_programmer_prompt_template.md
**Changes Made**:
- ✅ Updated architecture context to 6-layer design
- ✅ Added workspace-aware development references
- ✅ Enhanced refinement patterns and examples
- ✅ Updated registry integration patterns
- ✅ Added script testability refinement guidance

**Migration Impact**:
- Code refinement follows modern architectural patterns
- Workspace-aware refinement ensures proper isolation
- Enhanced patterns improve refinement quality

### Supporting Templates

#### 8. README.md
**Changes Made**:
- ✅ Updated agentic workflow to reflect 6-layer architecture
- ✅ Added workspace-aware development overview
- ✅ Enhanced agent descriptions with modern capabilities
- ✅ Updated knowledge base organization
- ✅ Added migration guidance references

**Migration Impact**:
- Documentation reflects current system architecture
- Developers understand modern workflow capabilities
- Clear guidance for using updated templates

#### 9. planner_prompt_template.md (Legacy Template)
**Changes Made**:
- ✅ Updated to 6-layer architecture description
- ✅ Added modern feature context
- ✅ Enhanced alignment rules
- ✅ Updated critical requirements

**Migration Impact**:
- Legacy template now compatible with modern architecture
- Maintains backward compatibility while encouraging modernization

#### 10. validator_prompt_template.md (Legacy Template)
**Changes Made**:
- ✅ Updated to 6-layer architecture description
- ✅ Added modern feature context
- ✅ Enhanced validation criteria
- ✅ Updated alignment rules

**Migration Impact**:
- Legacy validation template now covers modern patterns
- Maintains compatibility while encouraging modern practices

## Knowledge Base Updates

### New Knowledge Base Sections Added

#### Workspace-Aware Development References
- `slipbox/1_design/workspace_aware_system_master_design.md`
- `slipbox/01_developer_guide_workspace_aware/README.md`
- `slipbox/01_developer_guide_workspace_aware/ws_workspace_cli_reference.md`
- `slipbox/1_design/workspace_aware_multi_developer_management_design.md`
- `slipbox/01_developer_guide_workspace_aware/ws_hybrid_registry_integration.md`

#### Three-Tier Configuration Design
- `slipbox/0_developer_guide/three_tier_config_design.md`

#### Script Testability Implementation
- `slipbox/0_developer_guide/script_testability_implementation.md`

#### Enhanced Validation Framework
- `slipbox/1_design/two_level_alignment_validation_system_design.md`
- `slipbox/1_design/universal_step_builder_test.md`
- `slipbox/1_design/sagemaker_step_type_classification_design.md`

### Updated Registry References
- **Old**: `src/cursus/steps/registry/`
- **New**: `src/cursus/registry/step_names_original.py`
- **Reason**: Reflects actual STEP_NAMES dictionary location and UnifiedRegistryManager integration

## Developer Action Items

### Immediate Actions Required

#### 1. Update Development Environment
- [ ] Review workspace-aware development options
- [ ] Choose between shared workspace or isolated workspace development
- [ ] Set up appropriate workspace configuration
- [ ] Update CLI tools to workspace-aware versions

#### 2. Review Architecture Understanding
- [ ] Study 6-layer architecture documentation
- [ ] Understand three-tier configuration design
- [ ] Learn UnifiedRegistryManager patterns
- [ ] Review script testability implementation patterns

#### 3. Update Development Practices
- [ ] Adopt workspace-aware development workflows
- [ ] Use UnifiedRegistryManager for registry operations
- [ ] Implement three-tier configuration design in new components
- [ ] Follow script testability patterns for new scripts

### Gradual Migration Actions

#### 1. Existing Component Updates
- [ ] Gradually migrate existing components to three-tier configuration
- [ ] Update registry integration to use UnifiedRegistryManager patterns
- [ ] Refactor scripts to use unified main function interface
- [ ] Add workspace-aware validation to existing components

#### 2. Documentation Updates
- [ ] Update component documentation to reflect modern patterns
- [ ] Add workspace context to component descriptions
- [ ] Document three-tier configuration field categorization
- [ ] Update integration examples with modern patterns

#### 3. Testing Updates
- [ ] Update tests to use workspace-aware validation
- [ ] Add tests for three-tier configuration patterns
- [ ] Update integration tests for UnifiedRegistryManager
- [ ] Add script testability validation tests

## Compatibility and Backward Support

### Backward Compatibility
- **Legacy Templates**: Old templates still work but are deprecated
- **4-Layer References**: Still understood but should be updated to 6-layer
- **Registry Patterns**: Legacy patterns supported but UnifiedRegistryManager preferred
- **Configuration Patterns**: Legacy patterns work but three-tier design recommended

### Deprecation Timeline
- **Phase 1 (Current)**: Both legacy and modern patterns supported
- **Phase 2 (Future)**: Legacy patterns deprecated with warnings
- **Phase 3 (Future)**: Legacy patterns removed, modern patterns required

### Migration Support
- **Documentation**: Comprehensive migration guides available
- **Examples**: Modern implementation examples provided
- **Validation**: Enhanced validation helps identify migration opportunities
- **Tooling**: CLI tools support both legacy and modern patterns

## Common Migration Issues and Solutions

### Issue 1: Architecture Layer Confusion
**Problem**: Confusion between 4-layer and 6-layer architecture
**Solution**: 
- Review 6-layer architecture documentation
- Understand that Configuration Classes and Hyperparameters are now explicit layers
- Use modern templates that clearly describe 6-layer architecture

### Issue 2: Registry Integration Patterns
**Problem**: Using legacy registry patterns instead of UnifiedRegistryManager
**Solution**:
- Update registry references to `step_names_original.py`
- Use UnifiedRegistryManager patterns for registry operations
- Follow workspace-aware registry integration patterns

### Issue 3: Configuration Design Patterns
**Problem**: Not following three-tier configuration design
**Solution**:
- Categorize configuration fields as Essential/System/Derived
- Follow three-tier configuration implementation patterns
- Use enhanced validation to verify three-tier compliance

### Issue 4: Script Testability Patterns
**Problem**: Scripts not following unified main function interface
**Solution**:
- Implement parameterized main function patterns
- Use environment collection entry point patterns
- Follow 12-point script refactoring checklist

### Issue 5: Workspace Context Confusion
**Problem**: Not understanding workspace-aware development options
**Solution**:
- Choose appropriate workspace type (shared vs isolated)
- Follow workspace-specific development patterns
- Use workspace-aware validation and testing

## Validation and Testing

### Updated Validation Criteria
- **6-Layer Architecture Compliance**: Components must follow modern architecture
- **Workspace-Aware Implementation**: Components must handle workspace context
- **Three-Tier Configuration**: Configuration classes must use proper field categorization
- **UnifiedRegistryManager Integration**: Registry operations must use modern patterns
- **Script Testability**: Scripts must follow unified main function interface

### Enhanced Testing Framework
- **Two-Level Validation**: Combines LLM analysis with strict tool validation
- **Workspace-Aware Testing**: Tests respect workspace boundaries and contexts
- **Comprehensive Coverage**: Tests cover alignment, standardization, and integration
- **Pattern Validation**: Tests verify modern architectural pattern compliance

## Support and Resources

### Documentation Resources
- **Architecture Guide**: `slipbox/0_developer_guide/README.md`
- **Workspace Guide**: `slipbox/01_developer_guide_workspace_aware/README.md`
- **Design Documents**: `slipbox/1_design/` (various modern design documents)
- **Implementation Examples**: `src/cursus/` (modern implementation patterns)

### Migration Support
- **Migration Guide**: This document
- **Template Updates**: All templates updated with modern patterns
- **Validation Tools**: Enhanced validation helps identify migration needs
- **CLI Support**: Workspace-aware CLI tools for development

### Community and Feedback
- **Issue Reporting**: Use standard issue reporting mechanisms
- **Feature Requests**: Request additional migration support features
- **Documentation Improvements**: Suggest improvements to migration guidance
- **Pattern Sharing**: Share successful migration patterns with community

## Conclusion

The migration from 4-layer to 6-layer architecture with

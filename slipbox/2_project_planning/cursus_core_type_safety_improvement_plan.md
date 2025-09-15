---
tags:
  - project
  - planning
  - code_quality
  - type_safety
  - mypy
keywords:
  - type annotations
  - mypy errors
  - code quality improvement
  - type safety
  - static analysis
  - cursus core
  - refactoring plan
topics:
  - code quality improvement
  - type safety enhancement
  - mypy compliance
  - technical debt reduction
language: python
date of note: 2025-09-15
---

# Cursus Core Type Safety Improvement Plan

## Executive Summary

Based on mypy analysis of `src/cursus/core/`, this document outlines a comprehensive plan to improve type safety and code quality. The analysis revealed 225 errors across 28 files after configuring mypy to ignore mods-related packages, indicating significant opportunities for improvement in type annotations, error handling, and code reliability.

## Current State Analysis

### MyPy Analysis Results (Updated with mods exclusions)
- **Files Analyzed**: 35 source files
- **Files with Errors**: 28 files (80% error rate)
- **Total Errors**: 225 errors (after excluding mods-related imports)
- **Configuration Updated**: Added mypy overrides for `mods_workflow_core.*`, `mods_workflow_helper.*`, and `secure_ai_sandbox_workflow_python_sdk.*`
- **Error Categories**:
  - Missing type annotations: ~40% (90 errors)
  - Incompatible types/assignments: ~25% (56 errors)
  - None handling issues: ~20% (45 errors)
  - Unreachable code: ~10% (23 errors)
  - Import/module issues: ~5% (11 errors) - *Most mods-related imports now ignored*

### Most Critical Files (Priority 1)
1. **`contract_base.py`** - 25+ errors (None handling, type incompatibilities)
2. **`dynamic_template.py`** - 20+ errors (missing imports, type assignments)
3. **`config_base.py`** - 15+ errors (type annotations, Any returns)
4. **`builder_base.py`** - 15+ errors (type annotations, incompatible assignments)
5. **`specification_base.py`** - 12+ errors (unreachable code, type annotations)

## Improvement Strategy

### Phase 1: Foundation (Weeks 1-2)
**Goal**: Fix critical type safety issues and establish baseline

#### 1.1 Import and Module Issues
- **Priority**: Medium (Reduced - mods packages now ignored)
- **Effort**: 1-2 hours
- **Files**: `dynamic_template.py`, others with import errors
- **Actions**:
  - **Skip mods-related imports**: `mods_workflow_core.utils.constants` and similar imports are now ignored via pyproject.toml configuration
  - Resolve undefined name errors for cursus internal classes (`DynamicPipelineTemplate`, `PipelineAssembler`)
  - Add proper import statements for cursus internal modules only
  - Focus on legitimate import issues, not customized package dependencies

#### 1.2 None Handling Critical Issues
- **Priority**: Critical
- **Effort**: 8-10 hours
- **Files**: `contract_base.py`, `config_base.py`, `dynamic_template.py`
- **Actions**:
  - Fix incompatible default arguments (None vs expected types)
  - Add proper Optional type annotations
  - Implement null checks before attribute access
  - Fix "None has no attribute" errors

#### 1.3 Unreachable Code Cleanup
- **Priority**: Medium
- **Effort**: 4-6 hours
- **Files**: Multiple files with unreachable statements
- **Actions**:
  - Remove or fix unreachable code blocks
  - Simplify complex conditional logic
  - Fix logical errors causing unreachable statements

### Phase 2: Type Annotations (Weeks 3-4)
**Goal**: Add comprehensive type annotations to improve code clarity

#### 2.1 Function Signatures
- **Priority**: High
- **Effort**: 12-15 hours
- **Files**: All files with missing type annotations
- **Actions**:
  - Add return type annotations to ~30 functions
  - Add parameter type annotations to ~60 functions
  - Use `-> None` for functions that don't return values
  - Add proper generic types where applicable

#### 2.2 Variable Type Annotations
- **Priority**: Medium
- **Effort**: 6-8 hours
- **Files**: Files with "Need type annotation" errors
- **Actions**:
  - Add type hints for class attributes
  - Add type hints for complex variables (dicts, lists)
  - Use proper generic types (Dict[str, Any], List[str], etc.)

### Phase 3: Type Compatibility (Weeks 5-6)
**Goal**: Fix type incompatibility issues and improve type safety

#### 3.1 Assignment Compatibility
- **Priority**: High
- **Effort**: 10-12 hours
- **Files**: Files with incompatible assignment errors
- **Actions**:
  - Fix incompatible return value types
  - Resolve type mismatches in assignments
  - Add proper type casting where necessary
  - Fix generic type parameter issues

#### 3.2 Function Call Compatibility
- **Priority**: High
- **Effort**: 8-10 hours
- **Files**: Files with argument type errors
- **Actions**:
  - Fix incompatible argument types
  - Add missing required arguments
  - Fix method signature overrides
  - Resolve generic type constraints

### Phase 4: Advanced Type Safety (Weeks 7-8)
**Goal**: Implement advanced type safety patterns and best practices

#### 4.1 Generic Types and Protocols
- **Priority**: Medium
- **Effort**: 6-8 hours
- **Actions**:
  - Implement proper generic type constraints
  - Add Protocol definitions for interfaces
  - Use TypeVar for generic functions
  - Add proper bounds for generic types

#### 4.2 Union Types and Optional Handling
- **Priority**: Medium
- **Effort**: 8-10 hours
- **Actions**:
  - Replace Any returns with proper Union types
  - Implement proper Optional handling patterns
  - Add type guards for runtime type checking
  - Use Literal types where appropriate

## Implementation Plan by File

### Priority 1 Files (Critical)

#### `contract_base.py` (25+ errors)
**Issues**: None handling, incompatible assignments, unreachable code
**Estimated Effort**: 6-8 hours
**Key Fixes**:
- Fix Optional parameter defaults (lines 77-80)
- Implement proper None checks in methods
- Fix Set[str] return type issues
- Remove unreachable code blocks

#### `dynamic_template.py` (20+ errors)
**Issues**: Missing imports, type assignments, attribute access
**Estimated Effort**: 6-8 hours (Reduced - mods imports now ignored)
**Key Fixes**:
- **Skip mods imports**: `mods_workflow_core.utils.constants` import error now ignored via configuration
- Fix undefined name errors for cursus internal classes (`DynamicPipelineTemplate`, `PipelineAssembler`)
- Add proper type annotations for complex variables
- Fix incompatible type assignments
- Focus on legitimate type safety issues, not external package dependencies

#### `config_base.py` (15+ errors)
**Issues**: Type annotations, Any returns, incompatible assignments
**Estimated Effort**: 5-7 hours
**Key Fixes**:
- Add return type annotations
- Replace Any returns with proper types
- Fix Optional parameter handling
- Add proper type annotations for methods

#### `builder_base.py` (15+ errors)
**Issues**: Type annotations, incompatible assignments, missing arguments
**Estimated Effort**: 6-8 hours
**Key Fixes**:
- Add comprehensive type annotations
- Fix incompatible assignment issues
- Resolve missing argument errors
- Add proper generic types

### Priority 2 Files (Important)

#### `specification_base.py` (12+ errors)
**Estimated Effort**: 4-6 hours
**Key Fixes**:
- Remove unreachable code
- Fix method signature overrides
- Add proper Optional handling

#### `config_field_categorizer.py` (15+ errors)
**Estimated Effort**: 4-5 hours
**Key Fixes**:
- Fix object indexing issues
- Add proper type annotations
- Resolve isinstance argument errors

#### `dependency_resolver.py` (10+ errors)
**Estimated Effort**: 5-6 hours
**Key Fixes**:
- Fix typing.Any vs builtins.any confusion
- Add proper type annotations
- Fix sort key function compatibility

## Quality Assurance Plan

### Testing Strategy
1. **Unit Tests**: Ensure all type fixes don't break functionality
2. **Integration Tests**: Verify type safety improvements work end-to-end
3. **MyPy Validation**: Run mypy after each phase to track progress
4. **Regression Testing**: Ensure no functionality is lost during refactoring

### Success Metrics
- **Error Reduction**: Target 80% reduction in mypy errors (from 225 to <45)
- **File Coverage**: Achieve clean mypy status for Priority 1 files
- **Type Coverage**: 90%+ of functions should have complete type annotations
- **Code Quality**: Eliminate all critical type safety issues

### Risk Mitigation
1. **Incremental Changes**: Make small, focused changes to minimize risk
2. **Version Control**: Use feature branches for each phase
3. **Code Review**: Require review for all type annotation changes
4. **Rollback Plan**: Maintain ability to rollback changes if issues arise

## Resource Requirements

### Time Estimation
- **Phase 1**: 11-16 hours (Foundation - reduced due to mods import exclusions)
- **Phase 2**: 18-23 hours (Type Annotations)
- **Phase 3**: 18-22 hours (Type Compatibility)
- **Phase 4**: 14-18 hours (Advanced Type Safety)
- **Total**: 61-79 hours (8-10 working days)

### Skills Required
- Strong Python typing knowledge
- MyPy expertise
- Understanding of cursus architecture
- Refactoring experience

### Tools Needed
- MyPy static type checker
- IDE with type checking support
- Comprehensive test suite
- Code review tools

## Long-term Benefits

### Code Quality Improvements
1. **Early Error Detection**: Catch type-related bugs at development time
2. **Better Documentation**: Type hints serve as inline documentation
3. **IDE Support**: Enhanced autocomplete and error detection
4. **Refactoring Safety**: Type checking prevents breaking changes

### Developer Experience
1. **Reduced Debugging Time**: Fewer runtime type errors
2. **Better Code Navigation**: IDEs can provide better navigation
3. **Increased Confidence**: Type safety provides confidence in changes
4. **Onboarding**: New developers can understand code faster

### Maintenance Benefits
1. **Reduced Technical Debt**: Cleaner, more maintainable code
2. **API Stability**: Type contracts prevent accidental API changes
3. **Documentation**: Self-documenting code through type hints
4. **Testing**: Better test coverage through type-driven development

## MyPy Configuration Updates

The following configuration has been added to `pyproject.toml` to handle mods-related imports:

```toml
[[tool.mypy.overrides]]
module = [
    "mods_workflow_core.*",
    "mods_workflow_helper.*",
    "secure_ai_sandbox_workflow_python_sdk.*",
]
ignore_missing_imports = true
ignore_errors = true
```

This configuration suppresses mypy errors for all mods-related packages, allowing focus on actual cursus code quality issues rather than external package dependencies.

## Conclusion

This comprehensive plan addresses the 225 mypy errors found in `cursus/core` (after excluding mods-related imports) through a phased approach focusing on critical issues first. The investment of 8-10 working days will significantly improve code quality, reduce bugs, and enhance developer productivity.

The plan prioritizes high-impact fixes while maintaining system stability through incremental changes and comprehensive testing. Upon completion, the cursus core will have robust type safety, better documentation through type hints, and significantly reduced technical debt.

## Next Steps

1. **MyPy Configuration**: ✅ **COMPLETED** - Updated pyproject.toml to ignore mods-related imports
2. **Approval**: Get stakeholder approval for the revised improvement plan
3. **Resource Allocation**: Assign developer resources for implementation
4. **Timeline**: Establish specific dates for each phase
5. **Kickoff**: Begin with Phase 1 foundation improvements (focusing on cursus internal issues)
6. **Progress Tracking**: Set up regular progress reviews and metrics tracking

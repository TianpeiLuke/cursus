---
tags:
  - prompt_template
  - validation
  - report_format
  - output_specification
  - llm_agent
keywords:
  - validation report format
  - two-level validation output
  - structured validation results
  - alignment validation report
  - LLM output specification
topics:
  - validation report structure
  - output format specification
  - validation result presentation
  - developer communication format
language: markdown
date of note: 2025-08-09
---

# Two-Level Validation Report Format Specification

## Related Documents

### Design and Implementation
- [Two-Level Alignment Validation System Design](../../1_design/two_level_alignment_validation_system_design.md) - Complete system design and architecture
- [Two-Level Alignment Validation Implementation Plan](../../2_project_planning/2025-08-09_two_level_alignment_validation_implementation_plan.md) - Implementation roadmap and planning
- [Two-Level Validation Agent Prompt Template](two_level_validation_agent_prompt_template.md) - LLM prompt template that uses this format

### Supporting Documentation
- [Alignment Rules](../../0_developer_guide/alignment_rules.md) - Alignment validation rules and requirements
- [Validation Checklist](../../0_developer_guide/validation_checklist.md) - Comprehensive validation checklist
- [Common Pitfalls](../../0_developer_guide/common_pitfalls.md) - Common implementation pitfalls to avoid

## Overview

This document specifies the expected output format for the Two-Level Alignment Validation Agent. The format is designed to provide comprehensive, actionable validation results that combine strict tool validation with architectural understanding.

**Key Design Principles**:
- **Structured Information**: Clear sections for different types of validation results
- **Actionable Recommendations**: Specific fixes with exact locations and reasoning
- **Traceability**: Clear connection between tool results and interpretations
- **Developer-Friendly**: Format optimized for developer consumption and action
- **Comprehensive Coverage**: All aspects of validation covered systematically

## Complete Report Format Template

```markdown
# Two-Level Validation Report for [Component Name]

**Component Type**: [ProcessingStep/TrainingStep/EvaluationStep/etc.]
**Validation Date**: [YYYY-MM-DD HH:MM:SS UTC]
**Validation Agent Version**: [Version identifier]

## Executive Summary

**Overall Status**: ✅ PASS / ❌ FAIL / ⚠️ CONDITIONAL PASS
**Critical Issues**: [Number] critical issues found
**Pattern Consistency**: [High/Medium/Low] - [Brief explanation]
**Ready for Integration**: [Yes/No] - [Brief reasoning]

### Quick Action Items
1. [Most critical issue requiring immediate attention]
2. [Second most critical issue]
3. [Third most critical issue if applicable]

---

## Architectural Analysis Summary

### Detected Patterns
- **Primary Pattern**: [PatternName] (Confidence: [XX]%)
  - **Description**: [Brief description of what this pattern means]
  - **Validation Implications**: [How this affects validation approach]
  
- **Secondary Pattern**: [PatternName] (Confidence: [XX]%)
  - **Description**: [Brief description]
  - **Validation Implications**: [How this affects validation]

- **Additional Patterns**: [List any other detected patterns with confidence scores]

### Implementation Approach Analysis
**Design Intent**: [Description of the apparent design intent and implementation strategy]

**Component Relationships**: 
- **Script ↔ Contract**: [Analysis of script-contract relationship]
- **Contract ↔ Specification**: [Analysis of contract-spec relationship]
- **Specification ↔ Dependencies**: [Analysis of dependency patterns]
- **Builder ↔ Configuration**: [Analysis of builder-config relationship]

**Data Flow Pattern**: [Description of how data flows through the component]

**Configuration Strategy**: [How configuration is handled and propagated]

### Validation Strategy Selected
**Tools Invoked**: [List of validation tools selected based on analysis]
**Rationale**: [Why these specific tools were chosen]
**Pattern Context Provided**: [What architectural context was provided to tools]

---

## Strict Validation Results

### Tool 1: Script-Contract Validation
**Status**: ✅ PASS / ❌ FAIL
**Tool**: `validate_script_contract_strict`
**Issues Found**: [Number] strict issues
**Execution Time**: [X.X] seconds

#### Tool Configuration
- **Pattern Context**: [List of patterns provided to tool]
- **Validation Mode**: Strict (zero tolerance)
- **Analysis Scope**: [Scope of analysis performed]

#### Critical Errors Found
1. **PATH_NOT_USED** (Line [XX])
   - **Issue**: Contract path `/opt/ml/input/data/[path]` (logical name: `[name]`) not found in script
   - **Location**: [File path:Line number]
   - **Tool Confidence**: 100% (deterministic)

2. **ENV_VAR_NOT_ACCESSED** (Line [XX])
   - **Issue**: Required environment variable `[VAR_NAME]` not accessed in script
   - **Location**: [File path:Line number]
   - **Tool Confidence**: 100% (deterministic)

#### Warnings Found
1. **UNUSED_IMPORT** (Line [XX])
   - **Issue**: [Description]
   - **Location**: [File path:Line number]

### Tool 2: Contract-Specification Validation
**Status**: ✅ PASS / ❌ FAIL
**Tool**: `validate_contract_spec_strict`
**Issues Found**: [Number] strict issues
**Execution Time**: [X.X] seconds

#### Tool Configuration
- **Pattern Context**: [Specification patterns provided]
- **Validation Mode**: Strict (exact logical name matching)
- **Logical Name Matching**: Exact string matching required

#### Critical Errors Found
1. **LOGICAL_NAME_MISMATCH_INPUT** 
   - **Issue**: Specification dependency `[logical_name]` not found in contract inputs
   - **Expected in Contract**: `[logical_name]`
   - **Found in Contract**: `[actual_names_list]`
   - **Tool Confidence**: 100% (deterministic)

2. **LOGICAL_NAME_MISMATCH_OUTPUT**
   - **Issue**: Contract output `[logical_name]` not found in specification outputs
   - **Expected in Specification**: `[logical_name]`
   - **Found in Specification**: `[actual_names_list]`
   - **Tool Confidence**: 100% (deterministic)

### Tool 3: Specification-Dependencies Validation
**Status**: ✅ PASS / ❌ FAIL
**Tool**: `validate_spec_dependencies_strict`
**Issues Found**: [Number] strict issues
**Execution Time**: [X.X] seconds

#### Tool Configuration
- **Pipeline Context**: [Available pipeline steps and external dependencies]
- **Dependency Patterns**: [Dependency patterns detected and provided]
- **Resolution Mode**: Strict (all dependencies must be resolvable)

#### Critical Errors Found
1. **UNRESOLVABLE_PIPELINE_DEPENDENCY**
   - **Issue**: Pipeline dependency `[logical_name]` cannot be resolved
   - **Compatible Sources**: `[list_of_compatible_sources]`
   - **Available Sources**: `[list_of_available_sources]`
   - **Resolution Failure Reason**: [Specific reason why resolution failed]
   - **Tool Confidence**: 100% (deterministic)

2. **INVALID_EXTERNAL_DEPENDENCY**
   - **Issue**: External dependency `[logical_name]` configuration invalid
   - **Validation Errors**: `[list_of_validation_errors]`
   - **Required Configuration**: [What configuration is required]
   - **Tool Confidence**: 100% (deterministic)

### Tool 4: Builder-Configuration Validation
**Status**: ✅ PASS / ❌ FAIL
**Tool**: `validate_builder_config_strict`
**Issues Found**: [Number] strict issues
**Execution Time**: [X.X] seconds

#### Tool Configuration
- **Usage Patterns**: [Configuration usage patterns detected]
- **Field Access Analysis**: Enhanced AST analysis
- **Framework Handling**: [SageMaker framework considerations]

#### Critical Errors Found
1. **UNDECLARED_FIELD_ACCESS**
   - **Issue**: Builder accesses undeclared configuration field: `[field_name]`
   - **Access Locations**: [List of file:line locations where field is accessed]
   - **Declared Fields**: [List of fields declared in configuration]
   - **Tool Confidence**: 100% (deterministic)

2. **REQUIRED_FIELD_NOT_ACCESSED**
   - **Issue**: Required configuration field not accessed: `[field_name]`
   - **Field Type**: `[field_type]`
   - **Is Required**: True
   - **Framework Handled**: [Yes/No with explanation]
   - **Tool Confidence**: 100% (deterministic)

---

## Integrated Analysis

### Pattern-Based Issue Filtering

#### False Positives Identified
**Total Issues Filtered**: [Number] issues identified as false positives

1. **Issue Type**: [ISSUE_TYPE]
   - **Tool Result**: [What the strict tool reported]
   - **Pattern Context**: [Which architectural pattern makes this acceptable]
   - **Reasoning**: [Detailed explanation of why this is a false positive]
   - **Confidence**: [High/Medium/Low] confidence in false positive assessment

2. **Issue Type**: [ISSUE_TYPE]
   - **Tool Result**: [What the strict tool reported]
   - **Pattern Context**: [Which architectural pattern makes this acceptable]
   - **Reasoning**: [Detailed explanation]
   - **Confidence**: [High/Medium/Low] confidence in assessment

#### Remaining Critical Issues
**Total Critical Issues After Filtering**: [Number] issues require attention

1. **Issue Type**: [ISSUE_TYPE]
   - **Tool Source**: [Which tool identified this issue]
   - **Severity**: Critical
   - **Impact**: [Description of what will happen if not fixed]
   - **Pattern Validation**: [Confirmed as violation, not pattern variation]

### Pattern Validation Assessment

#### Pattern Consistency Analysis
- **Primary Pattern Consistency**: [High/Medium/Low]
  - **Assessment**: [Whether the primary pattern is consistently applied]
  - **Inconsistencies Found**: [List any inconsistencies in pattern application]
  - **Impact**: [Effect of inconsistencies on maintainability]

- **Cross-Component Pattern Alignment**: [High/Medium/Low]
  - **Assessment**: [Whether patterns are consistent across related components]
  - **Alignment Issues**: [Any cross-component pattern misalignments]

#### Framework Considerations
- **SageMaker Framework Handling**: [Analysis of framework-handled vs. explicit handling]
- **Framework Delegation Patterns**: [Assessment of framework delegation usage]
- **Configuration Propagation**: [Analysis of how configuration flows through framework]

### Cross-Component Alignment Analysis

#### Logical Name Consistency
**Status**: ✅ CONSISTENT / ❌ INCONSISTENT / ⚠️ PARTIALLY CONSISTENT

**Analysis**:
- **Script ↔ Contract**: [Specific analysis with examples]
- **Contract ↔ Specification**: [Specific analysis with examples]
- **Cross-Component Flow**: [Analysis of logical name flow across all components]

**Examples of Consistency**:
- ✅ Logical name `[name]` correctly flows: Spec → Contract → Script
- ✅ Logical name `[name]` correctly flows: Spec → Contract → Script

**Examples of Inconsistency**:
- ❌ Logical name `[name]` missing in: [Component where missing]
- ❌ Logical name mismatch: `[name1]` in Spec vs `[name2]` in Contract

#### Dependency Resolution Analysis
**Status**: ✅ ALL RESOLVABLE / ❌ UNRESOLVABLE DEPENDENCIES / ⚠️ PARTIAL RESOLUTION

**Pipeline Dependencies**:
- **Resolvable**: [List of dependencies that can be resolved with sources]
- **Unresolvable**: [List of dependencies that cannot be resolved with reasons]

**External Dependencies**:
- **Valid Configuration**: [List of external dependencies with valid configuration]
- **Invalid Configuration**: [List of external dependencies with configuration issues]

**Configuration Dependencies**:
- **Valid References**: [List of configuration dependencies with valid field references]
- **Invalid References**: [List of configuration dependencies with invalid references]

#### Configuration Flow Analysis
**Status**: ✅ PROPER FLOW / ❌ FLOW ISSUES / ⚠️ PARTIAL FLOW

**Configuration Propagation Path**:
1. **Config Class** → **Builder** → **SageMaker Step** → **Container Environment**
2. **Analysis**: [Description of how configuration flows through each stage]
3. **Issues Found**: [Any breaks or issues in the configuration flow]

**Field Access Patterns**:
- **Required Fields**: [Analysis of required field access patterns]
- **Optional Fields**: [Analysis of optional field access patterns]
- **Framework-Handled Fields**: [Analysis of fields handled by SageMaker framework]

#### Path Alignment Analysis
**Status**: ✅ ALIGNED / ❌ MISALIGNED / ⚠️ PARTIALLY ALIGNED

**Path Usage Consistency**:
- **Contract-Defined Paths**: [List of paths defined in contract]
- **Script-Used Paths**: [List of paths actually used in script]
- **Alignment Assessment**: [Analysis of path usage consistency]

**Environment Variable Consistency**:
- **Contract-Defined Variables**: [List of environment variables defined]
- **Script-Accessed Variables**: [List of environment variables accessed]
- **Alignment Assessment**: [Analysis of environment variable consistency]

---

## Recommendations

### Critical Issues (Must Fix)

#### Issue 1: [Issue Type and Brief Description]
**Priority**: 🔴 CRITICAL
**Impact**: [Description of runtime impact if not fixed]
**Tool Source**: [Which validation tool identified this]

**Problem Details**:
- **Location**: `[file_path]:[line_number]`
- **Current Code**: 
  ```python
  [exact current code that has the issue]
  ```
- **Issue Description**: [Detailed description of what's wrong]

**Required Fix**:
- **New Code**:
  ```python
  [exact code that should replace the current code]
  ```
- **Explanation**: [Why this fix is required and how it addresses the issue]
- **Verification**: [How to verify the fix works]

**Additional Changes Required**:
- **File**: `[other_file_path]` - [Description of related changes needed]
- **File**: `[another_file_path]` - [Description of related changes needed]

#### Issue 2: [Issue Type and Brief Description]
**Priority**: 🔴 CRITICAL
**Impact**: [Description of runtime impact if not fixed]
**Tool Source**: [Which validation tool identified this]

[Same detailed structure as Issue 1]

### Pattern Improvements (Should Consider)

#### Improvement 1: [Pattern Issue Description]
**Priority**: 🟡 MEDIUM
**Impact**: [Effect on maintainability and consistency]

**Current Pattern**:
- **Description**: [How the component currently implements the pattern]
- **Issues**: [What's inconsistent or suboptimal about current approach]
- **Examples**: [Specific code examples showing current pattern]

**Recommended Pattern**:
- **Description**: [How the pattern should be implemented]
- **Benefits**: [Why this approach is better]
- **Examples**: [Specific code examples showing recommended pattern]

**Implementation Steps**:
1. [Step 1 with specific actions]
2. [Step 2 with specific actions]
3. [Step 3 with specific actions]

#### Improvement 2: [Consistency Issue Description]
**Priority**: 🟡 MEDIUM
**Impact**: [Effect on maintainability and reliability]

**Inconsistency Found**:
- **Description**: [What inconsistency was found across components]
- **Components Affected**: [List of components with inconsistent implementations]
- **Examples**: [Specific examples of the inconsistency]

**Standardization Approach**:
- **Recommended Standard**: [What the standard approach should be]
- **Migration Path**: [How to migrate from current inconsistent state]
- **Benefits**: [Why standardization helps]

### Best Practices (Nice to Have)

#### Practice 1: [Code Quality Improvement]
**Priority**: 🟢 LOW
**Impact**: [Benefit to code quality and maintainability]

**Current Approach**:
- **Description**: [How something is currently implemented]
- **Limitations**: [What could be improved]

**Suggested Improvement**:
- **Description**: [What improvement is suggested]
- **Implementation**: [How to implement the improvement]
- **Benefit**: [Why this improvement is valuable]

#### Practice 2: [Documentation Enhancement]
**Priority**: 🟢 LOW
**Impact**: [Benefit to developer understanding and maintenance]

**Missing Documentation**:
- **What's Missing**: [What documentation is missing or insufficient]
- **Impact**: [How lack of documentation affects development]

**Recommended Addition**:
- **What to Add**: [Specific documentation to add]
- **Location**: [Where to add the documentation]
- **Content**: [What the documentation should contain]
- **Value**: [Why this documentation helps developers]

---

## Overall Assessment

### Validation Summary
- **Strict Validation Score**: [X]/4 tools passed
- **Critical Issues Count**: [Number] must-fix issues
- **Pattern Consistency**: [High/Medium/Low] with [brief explanation]
- **Architectural Alignment**: [Assessment of overall alignment with design patterns]

### Integration Readiness
**Ready for Integration**: ✅ YES / ❌ NO / ⚠️ CONDITIONAL

**Reasoning**: [Detailed explanation of integration readiness assessment]

**Conditions for Integration** (if conditional):
1. [Condition 1 that must be met]
2. [Condition 2 that must be met]
3. [Condition 3 that must be met]

### Confidence Assessment
**Overall Confidence**: [High/Medium/Low] confidence in validation results

**Confidence Factors**:
- **Pattern Detection**: [High/Medium/Low] - [Reasoning]
- **Tool Accuracy**: [High/Medium/Low] - [Reasoning]
- **False Positive Filtering**: [High/Medium/Low] - [Reasoning]
- **Architectural Understanding**: [High/Medium/Low] - [Reasoning]

**Limitations**:
- [Any limitations in the validation analysis]
- [Areas where additional validation might be needed]
- [Assumptions made during validation]

---

## Validation Metadata

### Execution Details
- **Validation Timestamp**: [YYYY-MM-DD HH:MM:SS UTC]
- **Total Execution Time**: [X.X] seconds
- **Validation Agent Version**: [Version identifier]
- **Tool Versions**: 
  - `validate_script_contract_strict`: [version]
  - `validate_contract_spec_strict`: [version]
  - `validate_spec_dependencies_strict`: [version]
  - `validate_builder_config_strict`: [version]

### Pattern Analysis Metadata
- **Pattern Detection Confidence**: [XX]% overall confidence in pattern detection
- **Primary Pattern Confidence**: [XX]% confidence in primary pattern identification
- **Pattern Validation Coverage**: [XX]% of component analyzed for patterns

### Issue Analysis Metadata
- **Total Issues Detected**: [Number] issues found by strict tools
- **False Positive Filter Rate**: [XX]% of issues filtered as false positives
- **Critical Issue Rate**: [XX]% of remaining issues classified as critical
- **Issue Resolution Confidence**: [High/Medium/Low] confidence in issue classification

### Component Analysis Coverage
- **Files Analyzed**: [Number] files analyzed
- **Lines of Code Analyzed**: [Number] lines analyzed
- **AST Nodes Processed**: [Number] AST nodes processed
- **Pattern Matches Found**: [Number] pattern matches identified

---

## Appendix

### Tool Output Details
[Raw tool outputs can be included here for debugging purposes if needed]

### Pattern Detection Details
[Detailed pattern detection results can be included here for reference]

### Cross-References
- **Related Components**: [List of related components that might be affected]
- **Dependency Chain**: [Components that depend on this component]
- **Impact Analysis**: [Components that might be impacted by changes]
```

## Format Usage Guidelines

### When to Use This Format
- **Primary Use**: All two-level validation agent outputs should follow this format
- **Consistency**: Ensures consistent reporting across all validation runs
- **Traceability**: Provides clear traceability from tool results to recommendations
- **Developer Experience**: Optimized for developer consumption and action

### Format Customization
- **Component-Specific Sections**: Add sections specific to component types when needed
- **Tool-Specific Details**: Include additional tool-specific information when relevant
- **Context-Specific Recommendations**: Adapt recommendation categories based on context
- **Audience Adaptation**: Adjust detail level based on intended audience

### Integration with Development Workflow
- **CI/CD Integration**: Format designed for automated parsing and action
- **Code Review Integration**: Structure supports code review processes
- **Issue Tracking Integration**: Format enables automatic issue creation
- **Documentation Integration**: Results can be integrated into project documentation

## Format Evolution

This format specification should be updated based on:
- **Developer Feedback**: Improvements based on developer usage and feedback
- **Tool Evolution**: Updates to accommodate new validation tools
- **Workflow Changes**: Adaptations to support evolving development workflows
- **Integration Requirements**: Changes to support new integration points
- **Reporting Needs**: Enhancements to support additional reporting requirements

The format represents the current best practices for validation reporting but should evolve with the system and development practices.

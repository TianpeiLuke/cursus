# ML Pipeline Development Workflow Prompts

This directory contains the specialized prompts used in our agentic ML pipeline development workflow. These prompts support an iterative development process focused on high-quality, compatible pipeline components.

## Workflow Overview

Our ML pipeline development follows a structured workflow using specialized AI agents:

1. **Planning Phase** (Initial Planner)
   - Takes requirements and creates an initial implementation plan
   - Incorporates architectural principles, design patterns, and integration considerations

2. **Plan Validation Phase** (Plan Validator)
   - Evaluates the plan for architectural correctness and alignment with standards
   - Identifies issues with special focus on compatibility with other pipeline components
   - Provides a detailed validation report

3. **Plan Revision Phase** (Revision Planner)
   - Takes validation feedback and produces an improved implementation plan
   - Addresses all identified issues and implements suggested improvements
   - This cycle can repeat until the plan passes validation

4. **Implementation Phase** (Programmer)
   - Takes the validated plan and creates the actual code
   - Follows the plan's architecture and implementation details precisely

5. **Code Validation Phase** (Validator)
   - Evaluates the implemented code against our architectural standards
   - Verifies that all requirements are met and integration works correctly
   - Provides a detailed validation report on the implementation

## Prompt Files

### [Initial Planner Prompt](initial_planner_prompt.md)
- **Purpose**: Create an initial implementation plan for a new pipeline step
- **Input**: Step requirements, architectural documentation
- **Output**: Comprehensive implementation plan with all required components
- **Key Focus**: Understanding requirements and designing an architecturally sound approach

### [Plan Validator Prompt](plan_validator_prompt.md)
- **Purpose**: Validate implementation plans against architectural standards
- **Input**: Implementation plan, architectural documentation
- **Output**: Detailed validation report with issues and recommendations
- **Key Focus**: Alignment rules, cross-component compatibility, standardization compliance

### [Revision Planner Prompt](revision_planner_prompt.md)
- **Purpose**: Update implementation plans based on validation feedback
- **Input**: Current implementation plan, validation report
- **Output**: Revised implementation plan addressing all issues
- **Key Focus**: Addressing compatibility issues, especially integration with other components

### [Programmer Prompt](programmer_prompt.md)
- **Purpose**: Implement code based on the validated implementation plan
- **Input**: Validated implementation plan, architectural documentation, example implementations
- **Output**: Complete code files in the correct project structure locations
- **Key Focus**: Following the plan precisely while ensuring alignment across components

### [Validator Prompt](validator_prompt.md)
- **Purpose**: Validate code implementation against architectural standards
- **Input**: Implementation code, implementation plan
- **Output**: Detailed validation report with issues and recommendations
- **Key Focus**: Verifying alignment across all components, cross-component compatibility

## Priority Assessment Areas

All validation prompts focus on these key areas, with special emphasis on:

1. **Alignment Rules Adherence** (40% weight)
   - Contract-to-specification alignment
   - Script-to-contract alignment
   - Builder-to-configuration alignment
   - Property path correctness

2. **Cross-Component Compatibility** (30% weight)
   - Dependency resolver compatibility scores
   - Output to input type matching
   - Logical name consistency
   - Semantic keyword effectiveness

3. **Standardization Rules Compliance** (30% weight)
   - Naming conventions
   - Interface standardization
   - Documentation standards
   - Error handling standards

## Example Usage

A typical workflow might proceed as:

1. Initial requirements provided to the Initial Planner
2. Implementation plan created and passed to Plan Validator
3. Validation report identifies issues in cross-component compatibility
4. Implementation plan sent to Revision Planner with validation report
5. Revised plan created with fixes for compatibility issues
6. Revised plan validated and approved by Plan Validator
7. Approved plan implemented by Programmer
8. Implementation validated by Validator
9. Any implementation issues fixed by Programmer
10. Final implementation approved for production use

This approach ensures high-quality, compatible pipeline components that integrate seamlessly into the existing architecture.

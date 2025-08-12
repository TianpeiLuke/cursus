# Agentic ML Pipeline Development Workflow Prompts

This directory contains the specialized prompt templates that power our **7-step agentic workflow** for automated ML pipeline step development. These prompts enable a multi-agent system with human-in-the-loop validation to ensure high-quality, compliant pipeline step implementations.

> **📋 Complete Design Reference**: See [Agentic Workflow Design](../../1_design/agentic_workflow_design.md) for the complete system architecture and detailed specifications.

## 🔄 Agentic Workflow Overview

Our agentic ML pipeline development system employs **four specialized AI agents** working in **two main phases** across **7 structured steps**:

### 🤖 Agent Roles

| Agent | Role | Color Code | Primary Function |
|-------|------|------------|------------------|
| 🎯 **Planner Agent** | Blue | Plan Creation & Revision | Creates and revises implementation plans |
| 🔍 **Validator Agent** | Purple | Quality Assurance | Validates plans and code with adaptive approaches:<br/>• **Plan Validation**: Level 1 only (LLM analysis)<br/>• **Code Validation**: Two-level (LLM + tools) |
| 💻 **Programmer Agent** | Green | Code Implementation | Generates and refines production-ready code |
| 👤 **Human-in-the-Loop** | Orange | Oversight & Guidance | Provides requirements, reviews, and approvals |

### 📊 Workflow Phases

**Phase 1: Plan Development & Validation** (Steps 1-3)
- Iterative plan creation and validation cycle
- Human guidance on requirements and documentation locations
- Convergence to a validated, implementable plan

**Phase 2: Code Implementation & Validation** (Steps 4-7)
- Code generation based on validated plan
- Two-level code validation with tool integration
- Iterative code refinement until validation passes

### 🗺️ Complete Workflow Diagram

```mermaid
flowchart TD
    %% User Input
    A[User Requirements<br/>Step Type & Documentation Locations] --> B[Step 1: Initial Planning]
    
    %% Phase 1: Plan Development and Validation
    subgraph Phase1 ["Phase 1: Plan Development & Validation"]
        B[Step 1: Initial Planning<br/>Agent: Planner<br/>Template: step1_initial_planner_prompt_template.md] --> C[Step 2: Plan Validation Cycle<br/>Agent: Validator<br/>Template: step2_plan_validator_prompt_template.md]
        
        C --> D{Validation<br/>Passed?}
        D -->|No| E[Step 3: Plan Revision<br/>Agent: Planner<br/>Template: step3_revision_planner_prompt_template.md]
        E --> C
        D -->|Yes| F[✓ Plan Convergence<br/>✓ Alignment Score ≥ 9/10<br/>✓ Standardization Score ≥ 8/10<br/>✓ Compatibility Score ≥ 8/10]
    end
    
    %% Phase 2: Code Implementation and Validation
    subgraph Phase2 ["Phase 2: Code Implementation & Validation"]
        F --> G[Step 4: Code Implementation<br/>Agent: Programmer<br/>Template: step4_programmer_prompt_template.md]
        
        G --> H[Step 5: Code Validation<br/>Agent: Validator<br/>Templates:<br/>• step5a_two_level_validation_agent_prompt_template.md<br/>• step5b_two_level_standardization_validation_agent_prompt_template.md]
        
        H --> I{Validation<br/>Passed?}
        I -->|No| J[Step 6: Code Refinement<br/>Agent: Programmer<br/>Template: step6_code_refinement_programmer_prompt_template.md]
        J --> K[Step 7: Validation Convergence<br/>Agent: Validator<br/>Repeat Step 5 Templates]
        K --> L{All Validations<br/>Pass?}
        L -->|No| J
        L -->|Yes| M[✅ Production Ready Implementation]
        I -->|Yes| M
    end
    
    %% Human-in-the-Loop Integration
    N[👤 Human Review & Approval] -.-> C
    N -.-> H
    N -.-> K
    N -.-> M
    
    %% Styling
    classDef plannerAgent fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef validatorAgent fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef programmerAgent fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    classDef userInput fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef decision fill:#ffecb3,stroke:#f57f17,stroke-width:2px
    classDef success fill:#c8e6c9,stroke:#2e7d32,stroke-width:3px
    classDef phase fill:#f5f5f5,stroke:#424242,stroke-width:2px,stroke-dasharray: 5 5
    
    class B,E plannerAgent
    class C,H,K validatorAgent
    class G,J programmerAgent
    class A,N userInput
    class D,I,L decision
    class M success
```

## 📝 Detailed Workflow Steps

### Step 1: Initial Planning 🎯
**Agent**: Planner | **Template**: `step1_initial_planner_prompt_template.md`
- **Input**: User requirements, step type categorization, documentation locations
- **Output**: Initial implementation plan with architectural design
- **Key Features**: Knowledge base integration, design pattern selection, alignment planning

### Step 2: Plan Validation Cycle 🔍
**Agent**: Validator | **Template**: `step2_plan_validator_prompt_template.md`
- **Input**: Implementation plan from Step 1
- **Output**: Validation report with scored assessment
- **Key Features**: **Level 1 validation only** (LLM-based analysis), compatibility analysis, standardization compliance
- **Process**: Iterative validation until convergence criteria met (Alignment ≥9/10, Standardization ≥8/10, Compatibility ≥8/10)
- **Important Note**: Plan validation uses Level 1 only because code-based validation tools cannot be applied without actual program code

### Step 3: Plan Revision 🎯
**Agent**: Planner | **Template**: `step3_revision_planner_prompt_template.md`
- **Input**: Validation report + original plan
- **Output**: Revised implementation plan addressing all issues
- **Key Features**: Issue-driven revision, architectural integrity maintenance
- **Process**: Cycles back to Step 2 until plan convergence is achieved

### Step 4: Code Implementation 💻
**Agent**: Programmer | **Template**: `step4_programmer_prompt_template.md`
- **Input**: Validated implementation plan
- **Output**: Complete code implementation (contracts, specs, builders, scripts)
- **Key Features**: Pattern-driven implementation, alignment enforcement

### Step 5: Code Validation 🔍
**Agent**: Validator | **Templates**: 
- `step5a_two_level_validation_agent_prompt_template.md` (alignment validation)
- `step5b_two_level_standardization_validation_agent_prompt_template.md` (standardization validation)
- **Input**: Generated code implementation
- **Output**: Two-level validation report with tool integration
- **Key Features**: LLM analysis + deterministic tool validation

### Step 6: Code Refinement 💻
**Agent**: Programmer | **Template**: `step6_code_refinement_programmer_prompt_template.md`
- **Input**: Validation report + original code
- **Output**: Refined code addressing all validation issues
- **Key Features**: Validation-driven fixes, pattern preservation

### Step 7: Validation Convergence ✅
**Agent**: Validator | **Process**: Repeat Steps 5-6 until all validations pass
- All tool-based validations pass
- No critical alignment violations
- Production readiness achieved

## 📋 Prompt Template Files

### Step 1: [Initial Planner Prompt Template](step1_initial_planner_prompt_template.md)
- **Agent**: Planner
- **Purpose**: Create an initial implementation plan for a new pipeline step
- **Input**: Step requirements, architectural documentation
- **Output**: Comprehensive implementation plan with all required components
- **Key Focus**: Understanding requirements and designing an architecturally sound approach

### Step 2: [Plan Validator Prompt Template](step2_plan_validator_prompt_template.md)
- **Agent**: Validator
- **Purpose**: Validate implementation plans against architectural standards (Level 1 validation only)
- **Input**: Implementation plan, architectural documentation
- **Output**: Detailed validation report with issues and recommendations
- **Key Focus**: Alignment rules, cross-component compatibility, standardization compliance

### Step 3: [Revision Planner Prompt Template](step3_revision_planner_prompt_template.md)
- **Agent**: Planner
- **Purpose**: Update implementation plans based on validation feedback
- **Input**: Current implementation plan, validation report
- **Output**: Revised implementation plan addressing all issues
- **Key Focus**: Addressing compatibility issues, especially integration with other components

### Step 4: [Programmer Prompt Template](step4_programmer_prompt_template.md)
- **Agent**: Programmer
- **Purpose**: Implement code based on the validated implementation plan
- **Input**: Validated implementation plan, architectural documentation, example implementations
- **Output**: Complete code files in the correct project structure locations
- **Key Focus**: Following the plan precisely while ensuring alignment across components

### Step 5a: [Two-Level Validation Agent Prompt Template](step5a_two_level_validation_agent_prompt_template.md)
- **Agent**: Validator
- **Purpose**: Validate code implementation using two-level validation (LLM + tools)
- **Input**: Implementation code, implementation plan
- **Output**: Comprehensive validation report with tool integration results
- **Key Focus**: Alignment validation with deterministic tool verification

### Step 5b: [Two-Level Standardization Validation Agent Prompt Template](step5b_two_level_standardization_validation_agent_prompt_template.md)
- **Agent**: Validator
- **Purpose**: Validate code standardization using two-level validation (LLM + tools)
- **Input**: Implementation code, standardization requirements
- **Output**: Standardization compliance report with tool results
- **Key Focus**: Naming conventions, interface standards, code quality

### Step 6: [Code Refinement Programmer Prompt Template](step6_code_refinement_programmer_prompt_template.md)
- **Agent**: Programmer
- **Purpose**: Refine code implementation based on validation feedback
- **Input**: Validation report, original code implementation
- **Output**: Refined code addressing all validation issues
- **Key Focus**: Validation-driven fixes while preserving architectural integrity

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

1. **Initial Planning**: Requirements provided to the Planner Agent
2. **Plan Validation Cycle**: Implementation plan validated iteratively until convergence
3. **Plan Revision**: Any issues addressed through revision cycles
4. **Code Implementation**: Approved plan implemented by Programmer Agent
5. **Code Validation**: Implementation validated using two-level validation approach
6. **Code Refinement**: Any validation issues fixed through refinement cycles
7. **Validation Convergence**: Final validation ensures production readiness

This streamlined approach ensures high-quality, compatible pipeline components that integrate seamlessly into the existing architecture while eliminating redundant validation steps.

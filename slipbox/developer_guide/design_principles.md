# Design Principles

Our pipeline architecture follows a set of core design principles that guide development and integration. Understanding these principles is essential for developing pipeline steps that align with the overall system design.

## Core Architectural Principles

These fundamental architectural principles form the foundation of our system design. They represent our highest-level design philosophies that influence all other principles and patterns.

### 1. Single Source of Truth

Centralize validation logic and configuration definitions in their respective component's configuration class to avoid redundancy and conflicts:

- **Centralized Configuration**: Each component owns its configuration definition and validation
- **Avoid Redundancy**: Don't duplicate validation or configuration logic across components
- **Clear Ownership**: Each component has clear ownership of its domain-specific knowledge
- **Consistent Access**: Provide consistent access patterns to configuration and validation

This principle is exemplified in several key architectural components:

- **Pipeline Registry**: A single registry for step names ensures uniqueness and consistency across the pipeline (see [`slipbox/pipeline_design/specification_registry.md`](../pipeline_design/specification_registry.md))
- **Standardization Rules**: Centralized rules for naming, path formatting, and environment variables (see [`slipbox/developer_guide/standardization_rules.md`](standardization_rules.md))
- **Alignment Rules**: A single source for validating alignment between specifications and implementations (see [`slipbox/developer_guide/alignment_rules.md`](alignment_rules.md))
- **Configuration Classes**: Each configuration class is the definitive source for its validation rules (see [`slipbox/pipeline_design/config.md`](../pipeline_design/config.md))

When you encounter configuration or validation logic, it should be defined exactly once, in the most appropriate component.

### 2. Declarative Over Imperative

Favor declarative specifications that describe *what* the pipeline should do rather than *how* to do it:

- **Define Intent**: Focus on defining what should happen, not implementation details
- **Configuration Over Code**: Use configuration to drive behavior rather than code
- **Separation of Definition and Execution**: Keep the definition of what should happen separate from how it happens
- **Self-Documenting**: Declarative definitions serve as documentation

This principle is the foundation of our specification-driven architecture:

- **Step Specifications**: Define dependencies and outputs declaratively, not how they're connected (see [`slipbox/pipeline_design/step_specification.md`](../pipeline_design/step_specification.md))
- **Script Contracts**: Declare expected paths and environment variables without implementation details (see [`slipbox/developer_guide/script_contract.md`](script_contract.md))
- **Configuration-Driven Pipeline Assembly**: Assemble pipelines through configuration rather than hardcoded steps (see [`slipbox/pipeline_design/specification_driven_design.md`](../pipeline_design/specification_driven_design.md))
- **DAG Definition**: Define pipeline structure declaratively without implementation details (see [`slipbox/pipeline_design/pipeline_dag.md`](../pipeline_design/pipeline_dag.md))

The step specification system exemplifies this by defining dependencies and outputs declaratively rather than through imperative code connections.

### 3. Type-Safe Specifications

Use strongly-typed enums and data structures (like `NodeType`, `DependencyType`) to prevent configuration errors at definition time:

- **Strong Typing**: Use enums and typed classes instead of strings and dictionaries
- **Compile-Time Checks**: Catch errors at definition time rather than runtime
- **IDE Support**: Enable IDE auto-completion and type checking
- **Self-Documenting**: Type definitions serve as documentation

We apply this principle throughout our architecture:

- **Dependency Resolution**: Strong typing for dependency types, ensuring compatibility (see [`slipbox/pipeline_design/dependency_resolution_explained.md`](../pipeline_design/dependency_resolution_explained.md))
- **Config Field Categorization**: Type-safe serialization and deserialization for configuration fields (see [`slipbox/pipeline_design/config_field_categorization_refactored.md`](../pipeline_design/config_field_categorization_refactored.md))
- **Pipeline Structure**: Typed node definitions (SOURCE, INTERNAL, SINK) that enforce structural rules (see [`slipbox/pipeline_design/pipeline_dag.md`](../pipeline_design/pipeline_dag.md))
- **Output Specifications**: Typed output specifications with explicit property paths (see [`slipbox/pipeline_design/step_specification.md`](../pipeline_design/step_specification.md))

By using strongly-typed specifications, we catch errors at definition time rather than runtime, improving robustness and developer experience.

### 4. Explicit Over Implicit

Favor explicitly defining connections and passing parameters between steps over implicit matching:

- **Named Connections**: Explicitly name connections between steps
- **Explicit Parameters**: Pass parameters explicitly rather than relying on naming conventions
- **Avoid Magic**: Don't rely on "magic" behavior or hidden conventions
- **Self-Documenting**: Explicit connections serve as documentation

This principle is evident throughout our system:

- **Step Specifications**: Explicit dependencies and outputs with clear logical names (see [`slipbox/pipeline_design/step_specification.md`](../pipeline_design/step_specification.md))
- **Script Contracts**: Explicitly defined input/output paths and environment variables (see [`slipbox/developer_guide/script_contract.md`](script_contract.md))
- **Property References**: Structured property path references that explicitly define data locations (see [`slipbox/pipeline_design/enhanced_property_reference.md`](../pipeline_design/enhanced_property_reference.md))
- **Semantic Keywords**: Explicit semantic matching criteria rather than implicit naming conventions (see [`slipbox/pipeline_design/dependency_resolution_explained.md`](../pipeline_design/dependency_resolution_explained.md))
- **Builder Mappings**: Explicit mapping from step types to builder classes (see [`slipbox/pipeline_design/step_builder.md`](../pipeline_design/step_builder.md))

When connections between components are explicit, the system becomes more maintainable, debuggable, and less prone to subtle errors. Our property reference system is a perfect example, where we explicitly define paths to properties rather than relying on implicit naming or position.

### Importance of Core Architectural Principles

These four core principles work together to create a robust, maintainable system:

- **Reduced Cognitive Load**: By following these principles, developers can understand one component at a time without needing to understand the entire system
- **Error Prevention**: Type safety and explicit connections prevent entire categories of errors
- **Maintainability**: Single source of truth and declarative specifications make the system easier to modify and extend
- **Debuggability**: Explicit connections and clear ownership make it easier to trace issues to their source
- **Documentation**: These principles produce self-documenting code that makes the system's intent clear

When these principles are followed consistently, the result is a system that is robust, maintainable, and adaptable to changing requirements.

### Cross-Influences Between Core Principles

These core principles reinforce and complement each other:

- **Single Source of Truth + Type-Safe Specifications**: Centralized configuration with strong typing ensures both uniqueness and correctness
- **Declarative Over Imperative + Explicit Over Implicit**: Declarative specifications require explicit connections to be useful and maintainable
- **Single Source of Truth + Explicit Over Implicit**: Explicit references to a single source prevent duplication and inconsistency
- **Type-Safe Specifications + Declarative Over Imperative**: Strong typing enhances the clarity and reliability of declarative specifications

Understanding these interconnections helps developers apply the principles consistently across the system.

## Design Principles

### 1. Separation of Concerns

Each component in the architecture has a specific, well-defined responsibility:

- **Step Specifications**: Define the "what" - inputs, outputs, and connectivity (see [`slipbox/pipeline_design/step_specification.md`](../pipeline_design/step_specification.md))
- **Script Contracts**: Define the "where" - container paths and environment variables (see [`slipbox/developer_guide/script_contract.md`](script_contract.md))
- **Step Builders**: Define the "how" - SageMaker integration and resources (see [`slipbox/pipeline_design/step_builder.md`](../pipeline_design/step_builder.md))
- **Processing Scripts**: Define the "logic" - business logic and algorithms (see [`slipbox/pipeline_design/specification_driven_design.md`](../pipeline_design/specification_driven_design.md))

This separation allows components to evolve independently while maintaining compatibility through well-defined interfaces.

This principle is strongly influenced by the **Single Source of Truth** and **Explicit Over Implicit** core principles, ensuring each component has clear ownership and explicit interfaces.

### 2. Specification-Driven Design

The architecture is fundamentally specification-driven, with specifications defining step requirements and capabilities:

- **Declarative Intent**: Express what you want, not how to implement it (see [`slipbox/pipeline_design/specification_driven_design.md`](../pipeline_design/specification_driven_design.md))
- **Explicit Contracts**: Make requirements and dependencies explicit (see [`slipbox/developer_guide/script_contract.md`](script_contract.md))
- **Self-Documenting**: Specifications serve as documentation (see [`slipbox/pipeline_design/step_specification.md`](../pipeline_design/step_specification.md))
- **Validation-First**: Validate at design time, not runtime (see [`slipbox/pipeline_design/environment_variable_contract_enforcement.md`](../pipeline_design/environment_variable_contract_enforcement.md))

By prioritizing specifications, we can ensure robustness, maintainability, and consistency across the pipeline.

This principle is a direct application of the **Declarative Over Imperative** and **Type-Safe Specifications** core principles, focusing on what the system should do rather than how it should do it, with strong typing to prevent errors.

### 3. Dependency Resolution via Semantic Matching

Dependencies between steps are resolved through semantic matching rather than hard-coded connections:

- **Logical Names**: Use descriptive names for inputs and outputs (see [`slipbox/pipeline_design/dependency_resolution_explained.md`](../pipeline_design/dependency_resolution_explained.md))
- **Semantic Keywords**: Enrich connections with semantic metadata (see [`slipbox/pipeline_design/dependency_resolver.md`](../pipeline_design/dependency_resolver.md))
- **Compatible Sources**: Explicitly define which steps can provide dependencies (see [`slipbox/pipeline_design/dependency_resolution_improvement.md`](../pipeline_design/dependency_resolution_improvement.md))
- **Required vs. Optional**: Clearly distinguish between required and optional dependencies (see [`slipbox/pipeline_design/dependency_resolution_summary.md`](../pipeline_design/dependency_resolution_summary.md))

This approach enables flexible pipeline assembly while maintaining strong validation.

This principle builds on the **Explicit Over Implicit** core principle by making connections between steps explicit through semantic metadata rather than implicit naming conventions. It also leverages **Type-Safe Specifications** through strongly-typed dependency definitions.

### 4. Build-Time Validation

Our architecture prioritizes catching issues at build time rather than runtime:

- **Contract Alignment**: Validate script contracts against specifications (see [`slipbox/developer_guide/alignment_rules.md`](alignment_rules.md))
- **Property Path Consistency**: Ensure consistent property paths in outputs (see [`slipbox/pipeline_design/enhanced_property_reference.md`](../pipeline_design/enhanced_property_reference.md))
- **Cross-Step Validation**: Validate connectivity between steps (see [`slipbox/pipeline_design/dependency_resolution_improvement.md`](../pipeline_design/dependency_resolution_improvement.md))
- **Configuration Validation**: Validate configurations before execution (see [`slipbox/pipeline_design/config.md`](../pipeline_design/config.md))

By shifting validation left, we reduce the risk of runtime failures and improve developer experience.

This principle is enabled by our **Type-Safe Specifications** core principle, which allows for compile-time checking, and **Single Source of Truth**, which ensures validation rules are consistently applied from a central definition.

### 5. Hybrid Design Approach

We follow a hybrid design approach that combines the best of specification-driven and config-driven approaches:

- **Specifications for Dependencies**: Use specifications for dependency resolution (see [`slipbox/pipeline_design/specification_driven_design.md`](../pipeline_design/specification_driven_design.md))
- **Configurations for Implementation**: Use configurations for SageMaker implementation details (see [`slipbox/pipeline_design/config_driven_design.md`](../pipeline_design/config_driven_design.md))
- **Universal Resolution Logic**: Apply consistent resolution across all pipeline types (see [`slipbox/pipeline_design/hybrid_design.md`](../pipeline_design/hybrid_design.md))
- **Progressive Interfaces**: Support both high-level and detailed interfaces (see [`slipbox/pipeline_design/fluent_api.md`](../pipeline_design/fluent_api.md))

This hybrid approach balances ease of use with flexibility and control.

This principle combines aspects of **Declarative Over Imperative** (for specifications) with pragmatic implementation concerns, allowing for clear separation between what the system should do and how it accomplishes it.

## Architectural Patterns

### Four-Layer Architecture

The pipeline follows a four-layer architecture:

1. **Specification Layer**: Defines step inputs, outputs, and connections (see [`slipbox/pipeline_design/step_specification.md`](../pipeline_design/step_specification.md))
2. **Contract Layer**: Defines script interface and environment (see [`slipbox/developer_guide/script_contract.md`](script_contract.md))
3. **Builder Layer**: Creates SageMaker steps and resolves dependencies (see [`slipbox/pipeline_design/step_builder.md`](../pipeline_design/step_builder.md))
4. **Script Layer**: Implements business logic (see [`slipbox/pipeline_design/specification_driven_design.md`](../pipeline_design/specification_driven_design.md))

Each layer has a well-defined responsibility and communicates with adjacent layers through explicit interfaces.

### Registry Pattern

Step components are registered in centralized registries:

- **Step Name Registry**: Maps step names to component types (see [`slipbox/pipeline_design/pipeline_registry.md`](../pipeline_design/pipeline_registry.md))
- **Specification Registry**: Provides access to step specifications (see [`slipbox/pipeline_design/specification_registry.md`](../pipeline_design/specification_registry.md))
- **Builder Registry**: Maps step types to builder classes (see [`slipbox/pipeline_design/registry_manager.md`](../pipeline_design/registry_manager.md))

This pattern enables discovery, validation, and consistency checking across the system.

The Registry Pattern is a practical implementation of the **Single Source of Truth** core principle, providing centralized access to component definitions and ensuring consistency across the system.

### Template Pattern

Pipeline templates provide reusable pipeline patterns:

- **DAG Definition**: Define the pipeline's directed acyclic graph (see [`slipbox/pipeline_design/pipeline_template_base.md`](../pipeline_design/pipeline_template_base.md))
- **Config Mapping**: Map configurations to steps (see [`slipbox/pipeline_design/pipeline_template_builder_v2.md`](../pipeline_design/pipeline_template_builder_v2.md))
- **Builder Mapping**: Map step types to builder classes (see [`slipbox/pipeline_design/pipeline_assembler.md`](../pipeline_design/pipeline_assembler.md))

Templates enforce consistent pipeline patterns while allowing customization through configurations.

The Template Pattern exemplifies the **Declarative Over Imperative** core principle by defining pipeline structures declaratively while allowing for customization through configuration rather than code changes.

## Implementation Principles

### Avoid Hardcoding

Avoid hardcoding paths, environment variables, or dependencies:

- **Use Script Contracts**: Reference paths from contracts
- **Use Specifications**: Reference dependencies from specifications
- **Use Configurations**: Reference parameters from configurations

Hardcoded values reduce flexibility and increase maintenance costs.

This principle directly supports the **Single Source of Truth** and **Explicit Over Implicit** core principles by ensuring references come from a single authoritative source through explicit paths rather than duplicated hardcoded values.

### Follow SageMaker Conventions

Adhere to SageMaker's conventions for container paths and environment:

- **Processing Inputs**: `/opt/ml/processing/input/{logical_name}`
- **Processing Outputs**: `/opt/ml/processing/output/{logical_name}`
- **Training Inputs**: `/opt/ml/input/data/{channel_name}`
- **Model Outputs**: `/opt/ml/model`

Following these conventions ensures compatibility with SageMaker's infrastructure.

### Test for Edge Cases

Always test for edge cases in your components:

- **Missing Dependencies**: How does your step handle missing optional dependencies?
- **Type Conversion**: Do you handle type conversion correctly in environment variables?
- **Path Handling**: Do you handle directory vs. file path differences?
- **Job Type Variants**: Does your step work for all job type variants?

Edge case testing improves robustness and reduces production issues.

### Design for Extensibility

Design your components with extensibility in mind:

- **Support Job Type Variants**: Allow for training, calibration, validation variants
- **Allow Configuration Override**: Make parameters configurable
- **Use Inheritance**: Leverage inheritance for shared functionality
- **Follow Template Method Pattern**: Define abstract methods for specialization

Extensible components adapt to changing requirements with minimal changes.

This principle builds on **Type-Safe Specifications** and **Declarative Over Imperative** by creating strongly-typed, configuration-driven extension points rather than requiring code changes for new variants or features.

## Design Anti-Patterns to Avoid

Each anti-pattern represents a violation of one or more core principles. Understanding which principles are violated helps explain why these patterns should be avoided.

### Anti-Pattern: Direct Script-to-Builder Coupling

**Violates**: Single Source of Truth, Explicit Over Implicit

**Avoid** having step builders directly reference script paths or environment variables without going through contracts:

```python
# WRONG - Hardcoded path
def _get_inputs(self):
    return [
        ProcessingInput(
            source=s3_uri,
            destination="/opt/ml/processing/input/data"  # Hardcoded
        )
    ]
```

**Correct** approach - use script contract:

```python
# CORRECT - Use contract
def _get_inputs(self, inputs):
    contract = self.spec.script_contract
    return [
        ProcessingInput(
            source=inputs["data"],
            destination=contract.expected_input_paths["data"]
        )
    ]
```

### Anti-Pattern: Property Path Inconsistency

**Violates**: Single Source of Truth, Type-Safe Specifications

**Avoid** inconsistent property paths in output specifications:

```python
# WRONG - Inconsistent property path
"output": OutputSpec(
    logical_name="output",
    property_path="properties.Outputs.output.S3Uri"  # Wrong format
)
```

**Correct** approach - use standard format:

```python
# CORRECT - Standard format
"output": OutputSpec(
    logical_name="output",
    property_path="properties.ProcessingOutputConfig.Outputs['output'].S3Output.S3Uri"
)
```

### Anti-Pattern: Script Path Hardcoding

**Violates**: Explicit Over Implicit, Single Source of Truth

**Avoid** hardcoding paths in scripts:

```python
# WRONG - Hardcoded paths
input_path = "/opt/ml/processing/input/data"  # Hardcoded
output_path = "/opt/ml/processing/output/data"  # Hardcoded
```

**Correct** approach - use contract enforcer:

```python
# CORRECT - Use contract enforcer
contract = get_script_contract()
with ContractEnforcer(contract) as enforcer:
    input_path = enforcer.get_input_path("data")
    output_path = enforcer.get_output_path("output")
```

### Anti-Pattern: Missing Script Contract Validation

**Violates**: Declarative Over Imperative, Type-Safe Specifications

**Avoid** deploying scripts without contract validation:

```python
# WRONG - No validation
def main():
    # Process data without validation
    process_data()
```

**Correct** approach - validate contract:

```python
# CORRECT - Validate contract
def main():
    contract = get_script_contract()
    validation = contract.validate_implementation(__file__)
    if not validation.is_valid:
        raise RuntimeError(f"Contract validation failed: {validation.errors}")
    process_data()
```

## Relationships Between Principles and Patterns

The following diagram illustrates how our principles and patterns interact:

```
Core Architectural Principles
├── Single Source of Truth ───────┐
│   └── Registry Pattern          │
├── Declarative Over Imperative ──┼─────┐
│   └── Template Pattern          │     │
├── Type-Safe Specifications ─────┼─────┼─────┐
│   └── Build-Time Validation     │     │     │
└── Explicit Over Implicit ───────┘     │     │
    └── Separation of Concerns          │     │
                                        │     │
Design Principles                       │     │
├── Separation of Concerns ─────────────┘     │
├── Specification-Driven Design ───────────────┘
├── Dependency Resolution via Semantic Matching
├── Build-Time Validation
└── Hybrid Design Approach

Implementation Principles
├── Avoid Hardcoding
├── Follow SageMaker Conventions
├── Test for Edge Cases
└── Design for Extensibility
```

This hierarchical relationship shows how our core architectural principles influence and inform our more specific design principles, patterns, and implementation guidelines. Understanding these relationships helps developers see the "why" behind each principle and pattern.

## Principle Application in the Development Lifecycle

Our principles apply throughout the development lifecycle:

1. **Design Phase**
   - Apply **Declarative Over Imperative** by defining specifications first
   - Use **Type-Safe Specifications** to create strongly-typed components
   - Follow **Separation of Concerns** to design clean component boundaries

2. **Implementation Phase**
   - Implement **Single Source of Truth** with centralized registries
   - Follow **Explicit Over Implicit** for component interfaces
   - Apply **Build-Time Validation** to catch issues early

3. **Testing Phase**
   - Test against **Type-Safe Specifications** to verify contract compliance
   - Verify **Explicit Over Implicit** connections between components
   - Test edge cases as guided by **Design for Extensibility**

4. **Maintenance Phase**
   - Leverage **Separation of Concerns** to make isolated changes
   - Use **Registry Pattern** to locate components needing modification
   - Follow **Avoid Hardcoding** to make changes in a single place

## Conclusion

By adhering to these design principles, we create a robust, maintainable pipeline architecture that supports a wide range of machine learning workflows. When developing new steps, ensure your implementation follows these principles to maintain consistency and quality across the system.

The cross-cutting nature of our core architectural principles (Single Source of Truth, Declarative Over Imperative, Type-Safe Specifications, and Explicit Over Implicit) provides a foundation that strengthens all aspects of the system design. By consistently applying these principles, we create a coherent architecture where components work together seamlessly while remaining independently maintainable.

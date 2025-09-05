# Detailed Component Guide

This guide provides detailed information about the key components involved in creating a new pipeline step. Each component plays a specific role in the architecture, and understanding how they fit together is crucial for successful integration.

## Overview of Components

The pipeline architecture follows a specification-driven approach with a six-layer design:

1. **Step Specifications**: Define inputs and outputs with logical names
2. **Script Contracts**: Define container paths for script inputs/outputs
3. **Processing Scripts**: Implement SageMaker-compatible business logic with unified interface
4. **Step Builders**: Connect specifications and contracts via SageMaker with workspace-aware registration
5. **Configuration Classes**: Define three-tier config structure with field management
6. **Hyperparameters**: Define model-specific configuration parameters (for training steps)

## Component Relationships

The components are related as follows:

- **Step Specifications** define how steps connect with other steps in the pipeline using logical names and provide S3 path information for pipeline connectivity
- **Script Contracts** define the interface between scripts and the SageMaker environment with container path information (where data should be placed/found inside containers)
- **Processing Scripts** implement the actual business logic and are executed in SageMaker containers
- **Step Builders** bridge specifications and contracts by creating SageMaker input/output objects (TrainingInput, ProcessingInput/Output, etc.) that map S3 sources to container destinations
- **Configuration Classes** define the three-tier config structure (Essential, System, Derived fields) with field management
- **Hyperparameters** (for training steps) define model-specific configuration parameters

### Input/Output Mapping Pattern

Step builders follow a consistent pattern across all step types:

1. **For SageMaker Input Objects** (TrainingInput, ProcessingInput, etc.):
   - **S3 source**: Provided by step specification dependencies
   - **Container destination**: Provided by script contract expected_input_paths

2. **For SageMaker Output Objects** (ProcessingOutput, etc.) or output parameters:
   - **Container source**: Provided by script contract expected_output_paths  
   - **S3 destination**: Provided by step specification outputs or generated from configuration

## Detailed Component Guides

For detailed guidance on developing each component, refer to the following sections:

- [Script Contract Development](script_contract.md): How to create and validate script contracts
- [Step Specification Development](step_specification.md): How to define step specifications for pipeline integration
- [Processing Script Development](script_development_guide.md): How to develop SageMaker-compatible scripts with unified interface
- [Step Builder Implementation](step_builder.md): How to implement the builder that creates SageMaker steps
- [Configuration Classes Development](three_tier_config_design.md): How to implement three-tier config design with field management
- [Config Field Manager Guide](config_field_manager_guide.md): How to use the config field manager for advanced field handling
- [Adding a New Hyperparameter Class](hyperparameter_class.md): How to create custom hyperparameter classes for training steps

## Component Alignment

The alignment between components is crucial for successful integration:

```mermaid
graph TD
    A[Step Specification] --> B[Step Builder]
    E[Configuration Classes] --> B
    A --> C[Script Contract]
    C --> D[Processing Script]
    E --> F[Hyperparameters]
    
    A -.->|contains| C
    A -.->|S3 sources/destinations| B
    C -.->|container paths| B
    C -.->|container paths| D
    B -.->|creates SageMaker objects| G[SageMaker Input/Output Objects]
    G -.->|maps S3 to container| D
    E -.->|field management| F
    B -.->|accesses via self.contract| C
    B -.->|accesses via self.spec| A
    
    subgraph "Input/Output Mapping"
        H[S3 Source<br/>from Specification] --> I[SageMaker Input Object]
        J[Container Destination<br/>from Contract] --> I
        I --> K[Script receives data<br/>at container path]
        
        L[Script writes data<br/>at container path] --> M[SageMaker Output Object]
        N[Container Source<br/>from Contract] --> M
        O[S3 Destination<br/>from Specification] --> M
    end
```

**4-Tier Alignment Validation System**:
1. **Script-Contract Alignment**: Scripts must use exactly the paths and arguments defined in contracts
2. **Contract-Specification Alignment**: Contract input/output paths must have matching logical names in specifications
3. **Specification-Dependency Alignment**: Step specification dependencies must match upstream steps' outputs
4. **Builder-Configuration Alignment**: Step builders must use configuration values correctly and register with UnifiedRegistryManager

**Key Alignment Requirements**:
- **Unified Main Function Interface**: All scripts must implement the standardized main function signature
- **SageMaker Container Compatibility**: Scripts must work with standard container paths
- **Three-Tier Config Structure**: Configuration classes must follow Essential/System/Derived field classification
- **Config Field Management**: Proper use of ConfigFieldManager for field derivation and validation
- **Workspace-Aware Registration**: Step builders must register with appropriate workspace context
- **Property Path Consistency**: All property paths must be validated across the entire chain

## Validation and Testing

Each component should be validated using our comprehensive validation framework:

1. **Processing Scripts**: 
   - Unit test the main function with various inputs
   - Integration test in simulated container environment
   - Validate unified main function interface compliance

2. **Script Contracts**: 
   - Validate against actual script implementation
   - Check path alignment and argument consistency
   - Verify environment variable requirements

3. **Step Specifications**: 
   - Validate property path consistency and contract alignment
   - Check dependency chain integrity
   - Verify logical name consistency

4. **Step Builders**: 
   - Test input/output generation and environment variable handling
   - Validate workspace-aware registration
   - Check SageMaker step configuration

5. **Configuration Classes**: 
   - Validate three-tier config structure (Essential/System/Derived)
   - Test config field manager functionality
   - Check field derivation and validation logic

6. **Integration**: 
   - Test end-to-end integration with other steps
   - Run 4-tier alignment validation
   - Validate workspace-aware functionality

**Validation Commands**:
```bash
# Validate step alignment (4-tier validation)
cursus validate-alignment --step YourStepType --workspace your_workspace

# Run comprehensive builder tests
cursus validate-builder --step YourStepType --workspace your_workspace

# Validate registry integration
cursus validate-registry --workspace your_workspace
```

For comprehensive validation guidance, see the [Validation Framework Guide](validation_framework_guide.md).

For more details on specific components, refer to the relevant sections linked above.

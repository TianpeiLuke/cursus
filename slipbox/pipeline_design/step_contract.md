# Step Contract

## What is the Purpose of Step Contract?

Step Contracts serve as **formal interface definitions** that establish clear, enforceable agreements between pipeline components. They represent the evolution from implicit assumptions to explicit, verifiable contracts in the ML pipeline architecture.

## Core Purpose

Step Contracts provide the **formal interface definition layer** that:

1. **Formal Interface Definition** - Explicit agreements between pipeline steps
2. **Design-Time Validation** - Compatibility checking during pipeline construction
3. **Runtime Contract Enforcement** - Ensure steps fulfill their promises
4. **Automatic Documentation Generation** - Living documentation synchronized with code
5. **Quality Gates and Governance** - Establish quality standards throughout pipeline

## Key Features

### 1. Formal Interface Definition

Step Contracts define explicit agreements between pipeline steps:

```python
@dataclass
class XGBoostTrainingContract(StepContract):
    # Input contracts
    required_inputs = {
        "training_data": DataContract(
            format="parquet",
            schema=TrainingDataSchema,
            quality_checks=[NoMissingValues(), ValidFeatureTypes()]
        ),
        "hyperparameters": ConfigContract(
            schema=XGBoostHyperparameters,
            validation_rules=[ValidTreeDepth(), ValidLearningRate()]
        )
    }
    
    # Output contracts
    guaranteed_outputs = {
        "model_artifacts": ModelContract(
            format="xgboost_model",
            metadata_includes=["feature_importance", "training_metrics"],
            quality_checks=[ModelValidation(), PerformanceThreshold(min_auc=0.7)]
        ),
        "training_metrics": MetricsContract(
            required_metrics=["accuracy", "precision", "recall", "auc"],
            format="json"
        )
    }
    
    # Behavioral contracts
    performance_guarantees = {
        "max_training_time": "2 hours",
        "memory_usage": "< 16GB",
        "reproducibility": "deterministic_with_seed"
    }
```

### 2. Design-Time Validation and Compatibility Checking

Enable early validation during pipeline design:

```python
def validate_pipeline_contracts(pipeline_steps):
    """Validate contracts during pipeline construction"""
    for i, step in enumerate(pipeline_steps[1:], 1):
        previous_step = pipeline_steps[i-1]
        
        # Check if outputs of previous step satisfy inputs of current step
        compatibility = check_contract_compatibility(
            previous_step.contract.guaranteed_outputs,
            step.contract.required_inputs
        )
        
        if not compatibility.is_compatible:
            raise ContractViolationError(
                f"Step {previous_step.name} outputs don't satisfy {step.name} inputs: "
                f"{compatibility.violations}"
            )
```

### 3. Runtime Contract Enforcement

Provide runtime validation to ensure steps fulfill their promises:

```python
class ContractEnforcedStep:
    def execute(self, inputs):
        # Pre-execution: Validate inputs against contract
        self.contract.validate_inputs(inputs)
        
        # Execute the actual step logic
        outputs = self._execute_internal(inputs)
        
        # Post-execution: Validate outputs against contract
        self.contract.validate_outputs(outputs)
        
        # Verify performance guarantees
        self.contract.validate_performance_metrics(self.execution_metrics)
        
        return outputs
```

### 4. Automatic Documentation Generation

Serve as living documentation that stays synchronized with code:

```python
def generate_contract_documentation(contract):
    """Auto-generate documentation from contracts"""
    doc = f"""
    # {contract.step_name} Step Contract
    
    ## Input Requirements
    """
    
    for input_name, input_contract in contract.required_inputs.items():
        doc += f"""
        ### {input_name} ({'required' if input_contract.required else 'optional'})
        - **Format**: {input_contract.format}
        - **Schema**: {input_contract.schema.__name__}
        - **Quality Checks**: {[check.__class__.__name__ for check in input_contract.quality_checks]}
        """
    
    doc += """
    ## Output Guarantees
    """
    
    for output_name, output_contract in contract.guaranteed_outputs.items():
        doc += f"""
        ### {output_name}
        - **Format**: {output_contract.format}
        - **Quality Guarantees**: {[check.__class__.__name__ for check in output_contract.quality_checks]}
        """
    
    return doc
```

### 5. Version Management and Backward Compatibility

Enable versioned interfaces and compatibility management:

```python
@dataclass
class XGBoostTrainingContractV2(StepContract):
    version = "2.0"
    backward_compatible_with = ["1.0", "1.1"]
    
    # New optional input in v2
    optional_inputs = {
        "validation_data": DataContract(
            format="parquet",
            schema=ValidationDataSchema,
            required=False  # Optional in v2, maintains v1 compatibility
        )
    }
    
    # Enhanced outputs in v2
    guaranteed_outputs = {
        **XGBoostTrainingContractV1.guaranteed_outputs,
        "validation_metrics": MetricsContract(  # New in v2
            required_metrics=["val_accuracy", "val_auc"],
            format="json"
        )
    }
```

## Contract Types

### 1. Data Contracts

Define data format, schema, and quality requirements:

```python
@dataclass
class DataContract:
    format: str  # "parquet", "csv", "json"
    schema: Type  # Pydantic or dataclass schema
    quality_checks: List[QualityCheck]
    size_constraints: Optional[Dict] = None
    
    def validate(self, data):
        """Validate data against contract"""
        # Check format
        if not self._validate_format(data):
            raise ContractViolationError(f"Data format must be {self.format}")
        
        # Check schema
        if not self._validate_schema(data):
            raise ContractViolationError("Data doesn't match required schema")
        
        # Run quality checks
        for check in self.quality_checks:
            if not check.validate(data):
                raise ContractViolationError(f"Quality check failed: {check}")
```

### 2. Model Contracts

Define model format, metadata, and performance requirements:

```python
@dataclass
class ModelContract:
    format: str  # "xgboost_model", "pytorch_model", "sklearn_model"
    metadata_includes: List[str]  # Required metadata fields
    quality_checks: List[ModelQualityCheck]
    performance_thresholds: Dict[str, float]
    
    def validate(self, model_artifacts):
        """Validate model against contract"""
        # Check format
        if not self._validate_model_format(model_artifacts):
            raise ContractViolationError(f"Model format must be {self.format}")
        
        # Check metadata
        for metadata_field in self.metadata_includes:
            if metadata_field not in model_artifacts.metadata:
                raise ContractViolationError(f"Missing required metadata: {metadata_field}")
        
        # Run quality checks
        for check in self.quality_checks:
            if not check.validate(model_artifacts):
                raise ContractViolationError(f"Model quality check failed: {check}")
```

### 3. Configuration Contracts

Define configuration schema and validation rules:

```python
@dataclass
class ConfigContract:
    schema: Type  # Configuration schema
    validation_rules: List[ValidationRule]
    required_fields: List[str]
    
    def validate(self, config):
        """Validate configuration against contract"""
        # Check required fields
        for field in self.required_fields:
            if not hasattr(config, field) or getattr(config, field) is None:
                raise ContractViolationError(f"Required field missing: {field}")
        
        # Run validation rules
        for rule in self.validation_rules:
            if not rule.validate(config):
                raise ContractViolationError(f"Validation rule failed: {rule}")
```

## Integration with Other Components

### With Step Specifications

Step Contracts complement and extend [Step Specifications](step_specification.md):

```python
# Step Specification defines the structural interface
DATA_PROCESSING_SPEC = StepSpecification(
    step_type="DataProcessing",
    node_type=NodeType.INTERNAL,
    dependencies=[...],
    outputs=[...]
)

# Step Contract defines the behavioral and quality interface
DATA_PROCESSING_CONTRACT = StepContract(
    specification=DATA_PROCESSING_SPEC,
    input_quality_requirements={...},
    output_quality_guarantees={...},
    performance_guarantees={...}
)
```

### With Quality Gates

Step Contracts establish quality gates throughout the pipeline:

```python
class QualityGate:
    def __init__(self, step_contract):
        self.contract = step_contract
    
    def validate_step_execution(self, step_instance, inputs, outputs):
        violations = []
        
        # Check input quality
        input_violations = self.contract.validate_input_quality(inputs)
        violations.extend(input_violations)
        
        # Check output quality
        output_violations = self.contract.validate_output_quality(outputs)
        violations.extend(output_violations)
        
        # Check performance guarantees
        perf_violations = self.contract.validate_performance(step_instance.metrics)
        violations.extend(perf_violations)
        
        if violations:
            raise QualityGateFailure(f"Step failed quality checks: {violations}")
```

### With Step Builders

Step Contracts work with [Step Builders](step_builder.md) to ensure implementation compliance:

```python
class ContractValidatedStepBuilder(BuilderStepBase):
    def __init__(self, config, contract):
        super().__init__(config)
        self.contract = contract
    
    def build_step(self, inputs):
        # Validate inputs against contract before building
        self.contract.validate_inputs(inputs)
        
        # Build step using parent implementation
        step = super().build_step(inputs)
        
        # Ensure step can fulfill contract guarantees
        self.contract.validate_step_capabilities(step)
        
        return step
```

### With Standardization Rules

Step Contracts enforce [standardization rules](standardization_rules.md) at the interface level:

```python
class StandardizedStepContract(StepContract):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Enforce standardization rules
        self._validate_naming_conventions()
        self._validate_interface_standards()
        self._validate_documentation_requirements()
    
    def _validate_naming_conventions(self):
        """Ensure contract follows naming standards"""
        for input_name in self.required_inputs.keys():
            if not re.match(r'^[a-z][a-z0-9_]*$', input_name):
                raise StandardizationError(f"Input name '{input_name}' violates naming convention")
```

### Cross-Team Collaboration

Enable clear team boundaries and parallel development:

```python
# Team A defines the contract for their data processing step
class DataProcessingContract(StepContract):
    guaranteed_outputs = {
        "processed_features": DataContract(
            schema=ProcessedFeatureSchema,
            format="parquet",
            quality_checks=[FeatureValidation(), DataDriftCheck()]
        )
    }

# Team B can develop against the contract without waiting for implementation
class ModelTrainingStep:
    def __init__(self):
        # Can validate compatibility during development
        assert is_compatible(
            DataProcessingContract.guaranteed_outputs["processed_features"],
            self.contract.required_inputs["training_data"]
        )
```

## Test Generation and Validation

Enable automatic test generation from contracts:

```python
def generate_contract_tests(step_contract):
    """Auto-generate tests from step contracts"""
    
    # Generate input validation tests
    for input_name, input_contract in step_contract.required_inputs.items():
        yield create_input_validation_test(input_name, input_contract)
    
    # Generate output validation tests  
    for output_name, output_contract in step_contract.guaranteed_outputs.items():
        yield create_output_validation_test(output_name, output_contract)
    
    # Generate performance tests
    for metric, threshold in step_contract.performance_guarantees.items():
        yield create_performance_test(metric, threshold)

# Usage
xgboost_tests = list(generate_contract_tests(XGBoostTrainingContract))
```

## Strategic Value

Step Contracts provide:

1. **Reliability**: Explicit contracts prevent integration failures
2. **Maintainability**: Clear interfaces make changes safer
3. **Testability**: Contracts enable comprehensive automated testing
4. **Scalability**: Teams can work independently against stable contracts
5. **Quality**: Built-in quality gates ensure pipeline reliability
6. **Documentation**: Self-documenting interfaces that stay current
7. **Governance**: Enforceable standards across the organization

## Example Usage

```python
# Define a complete step contract
training_contract = XGBoostTrainingContract(
    required_inputs={
        "training_data": DataContract(
            format="parquet",
            schema=TrainingDataSchema,
            quality_checks=[NoMissingValues(), ValidFeatureTypes()]
        )
    },
    guaranteed_outputs={
        "model_artifacts": ModelContract(
            format="xgboost_model",
            quality_checks=[ModelValidation(), PerformanceThreshold(min_auc=0.7)]
        )
    },
    performance_guarantees={
        "max_training_time": "2 hours",
        "memory_usage": "< 16GB"
    }
)

# Use contract for validation
try:
    training_contract.validate_inputs(inputs)
    outputs = training_step.execute(inputs)
    training_contract.validate_outputs(outputs)
except ContractViolationError as e:
    logger.error(f"Contract violation: {e}")
    raise
```

Step Contracts represent the **maturation of the pipeline architecture** from ad-hoc connections to formal, verifiable interfaces that enable enterprise-scale ML pipeline development with built-in quality assurance and governance.

---
tags:
  - design
  - step_builders
  - conditional_steps
  - patterns
  - sagemaker
  - control_flow
keywords:
  - conditional step patterns
  - ConditionStep
  - control flow
  - specification driven
  - dependency resolution
  - pipeline branching
topics:
  - step builder patterns
  - conditional step implementation
  - SageMaker conditional steps
  - pipeline control flow
language: python
date of note: 2025-10-22
---

# Conditional Step Builder Patterns

## Overview

This document analyzes the requirements and design patterns for implementing Conditional step builders in the cursus framework. Conditional steps enable pipeline branching and control flow based on runtime conditions, integrating seamlessly with the existing specification-driven dependency resolution system.

## SageMaker ConditionalStep Requirements

Based on AWS SageMaker documentation, ConditionalStep (also known as ConditionStep) provides the following capabilities:

### Core Components
- **Condition**: A condition object that evaluates to true/false at runtime
- **If Steps**: List of steps to execute when condition evaluates to true
- **Else Steps**: Optional list of steps to execute when condition evaluates to false
- **Dependencies**: Steps that must complete before condition evaluation

### Condition Types
1. **Property-Based Conditions**: Evaluate step properties against threshold values
   - `ConditionGreaterThan`, `ConditionGreaterThanOrEqualTo`
   - `ConditionLessThan`, `ConditionLessThanOrEqualTo`
   - `ConditionEquals`, `ConditionNotEquals`

2. **Function-Based Conditions**: Evaluate Lambda function outputs
   - `ConditionEquals` with Lambda step output

3. **Composite Conditions**: Combine multiple conditions with logical operators
   - `condition1.and_(condition2)`
   - `condition1.or_(condition2)`

### SageMaker Step Type Classification
Conditional steps create **ConditionStep** instances that control pipeline execution flow without producing traditional outputs like Processing or Training steps.

## Conditional Step Builder Patterns

### 1. Base Architecture Pattern

All Conditional step builders follow this consistent architecture:

```python
@register_builder()
class ConditionalStepBuilder(StepBuilderBase):
    def __init__(self, config, sagemaker_session=None, role=None, notebook_root=None, 
                 registry_manager=None, dependency_resolver=None):
        # Load conditional step specification
        spec = CONDITIONAL_STEP_SPEC
        super().__init__(config=config, spec=spec, ...)
        
    def validate_configuration(self) -> None:
        # Validate required conditional configuration
        
    def _build_condition(self, condition_input) -> Condition:
        # Create SageMaker condition object
        
    def _resolve_step_references(self, step_names) -> List[Step]:
        # Resolve step name references to actual step instances
        
    def _get_inputs(self, inputs) -> Dict[str, Any]:
        # Extract condition inputs using specification
        
    def _get_outputs(self, outputs) -> Dict[str, Any]:
        # Conditional steps typically don't produce outputs
        
    def create_step(self, **kwargs) -> ConditionStep:
        # Orchestrate conditional step creation
```

### 2. Property-Based Conditional Step Pattern

The most common pattern for conditional steps based on step property evaluation:

```python
class PropertyConditionalStepBuilder(StepBuilderBase):
    """
    Builder for conditional steps based on step property values.
    Example: Execute different paths based on model accuracy threshold.
    """
    
    def __init__(self, config, **kwargs):
        spec = PROPERTY_CONDITIONAL_SPEC
        super().__init__(config=config, spec=spec, **kwargs)
        
    def validate_configuration(self) -> None:
        required_attrs = [
            'condition_type', 'property_path', 'threshold_value',
            'comparison_operator', 'if_step_names'
        ]
        
        for attr in required_attrs:
            if not hasattr(self.config, attr) or getattr(self.config, attr) is None:
                raise ValueError(f"Missing required attribute: {attr}")
                
        # Validate comparison operator
        valid_operators = ['greater_than', 'greater_than_or_equal', 'less_than', 
                          'less_than_or_equal', 'equals', 'not_equals']
        if self.config.comparison_operator not in valid_operators:
            raise ValueError(f"Invalid comparison operator: {self.config.comparison_operator}")
    
    def _build_condition(self, condition_input: Any) -> Condition:
        """Build SageMaker condition from config and input."""
        from sagemaker.workflow.conditions import (
            ConditionGreaterThan, ConditionGreaterThanOrEqualTo,
            ConditionLessThan, ConditionLessThanOrEqualTo,
            ConditionEquals, ConditionNotEquals
        )
        
        operator_map = {
            'greater_than': ConditionGreaterThan,
            'greater_than_or_equal': ConditionGreaterThanOrEqualTo,
            'less_than': ConditionLessThan,
            'less_than_or_equal': ConditionLessThanOrEqualTo,
            'equals': ConditionEquals,
            'not_equals': ConditionNotEquals
        }
        
        condition_class = operator_map[self.config.comparison_operator]
        return condition_class(
            left=condition_input,
            right=self.config.threshold_value
        )
    
    def create_step(self, **kwargs) -> ConditionStep:
        # Extract inputs using dependency resolver
        dependencies = kwargs.get('dependencies', [])
        inputs = {}
        
        if dependencies and self.spec:
            extracted_inputs = self.extract_inputs_from_dependencies(dependencies)
            inputs.update(extracted_inputs)
        
        # Get condition input
        condition_input = inputs.get('condition_input')
        if not condition_input:
            raise ValueError("Required condition_input not resolved from dependencies")
        
        # Build condition
        condition = self._build_condition(condition_input)
        
        # Resolve step references (handled by pipeline assembler)
        if_steps = kwargs.get('if_steps', [])
        else_steps = kwargs.get('else_steps', [])
        
        return ConditionStep(
            name=self._get_step_name(),
            conditions=[condition],
            if_steps=if_steps,
            else_steps=else_steps,
            depends_on=dependencies
        )
```

### 3. Multi-Condition Conditional Step Pattern

For complex conditional logic with multiple conditions:

```python
class MultiConditionalStepBuilder(StepBuilderBase):
    """
    Builder for conditional steps with multiple conditions and logical operators.
    Example: Execute paths based on multiple model metrics.
    """
    
    def validate_configuration(self) -> None:
        if not hasattr(self.config, 'conditions') or not self.config.conditions:
            raise ValueError("Multi-conditional step requires 'conditions' list")
            
        for i, condition_config in enumerate(self.config.conditions):
            required_fields = ['input_name', 'comparison_operator', 'threshold_value']
            for field in required_fields:
                if field not in condition_config:
                    raise ValueError(f"Condition {i} missing required field: {field}")
    
    def _build_conditions(self, resolved_inputs: Dict[str, Any]) -> List[Condition]:
        """Build multiple conditions from config and inputs."""
        conditions = []
        
        for condition_config in self.config.conditions:
            input_name = condition_config['input_name']
            condition_input = resolved_inputs.get(input_name)
            
            if condition_input is None:
                if condition_config.get('required', True):
                    raise ValueError(f"Required condition input '{input_name}' not found")
                continue
                
            # Create individual condition
            condition = self._build_single_condition(condition_config, condition_input)
            conditions.append(condition)
            
        return conditions
    
    def _combine_conditions(self, conditions: List[Condition]) -> Condition:
        """Combine multiple conditions with logical operators."""
        if len(conditions) == 1:
            return conditions[0]
            
        # Apply logical operator from config
        logic_operator = getattr(self.config, 'condition_logic', 'AND')
        
        combined = conditions[0]
        for condition in conditions[1:]:
            if logic_operator == 'AND':
                combined = combined.and_(condition)
            elif logic_operator == 'OR':
                combined = combined.or_(condition)
            else:
                raise ValueError(f"Unsupported logic operator: {logic_operator}")
                
        return combined
    
    def create_step(self, **kwargs) -> ConditionStep:
        dependencies = kwargs.get('dependencies', [])
        
        # Extract multiple condition inputs
        resolved_inputs = {}
        if dependencies and self.spec:
            resolved_inputs = self.extract_inputs_from_dependencies(dependencies)
        
        # Build and combine conditions
        conditions = self._build_conditions(resolved_inputs)
        if not conditions:
            raise ValueError("No valid conditions could be built")
            
        final_condition = self._combine_conditions(conditions)
        
        # Get step references
        if_steps = kwargs.get('if_steps', [])
        else_steps = kwargs.get('else_steps', [])
        
        return ConditionStep(
            name=self._get_step_name(),
            conditions=[final_condition],
            if_steps=if_steps,
            else_steps=else_steps,
            depends_on=dependencies
        )
```

### 4. Function-Based Conditional Step Pattern

For conditional steps based on Lambda function evaluation:

```python
class FunctionConditionalStepBuilder(StepBuilderBase):
    """
    Builder for conditional steps based on Lambda function outputs.
    Example: Execute paths based on custom business logic evaluation.
    """
    
    def validate_configuration(self) -> None:
        required_attrs = ['function_output_value', 'comparison_operator']
        
        for attr in required_attrs:
            if not hasattr(self.config, attr):
                raise ValueError(f"Missing required attribute: {attr}")
    
    def create_step(self, **kwargs) -> ConditionStep:
        dependencies = kwargs.get('dependencies', [])
        
        # Extract Lambda function output
        resolved_inputs = {}
        if dependencies and self.spec:
            resolved_inputs = self.extract_inputs_from_dependencies(dependencies)
        
        lambda_output = resolved_inputs.get('lambda_output')
        if not lambda_output:
            raise ValueError("Required lambda_output not resolved from dependencies")
        
        # Create condition based on function output
        from sagemaker.workflow.conditions import ConditionEquals
        
        condition = ConditionEquals(
            left=lambda_output,
            right=self.config.function_output_value
        )
        
        # Get step references
        if_steps = kwargs.get('if_steps', [])
        else_steps = kwargs.get('else_steps', [])
        
        return ConditionStep(
            name=self._get_step_name(),
            conditions=[condition],
            if_steps=if_steps,
            else_steps=else_steps,
            depends_on=dependencies
        )
```

## Data Structure Extensions

### 1. Step Specification Extensions

Conditional steps require new dependency and output types:

```python
# Add new dependency types for conditional logic
class DependencyType(Enum):
    # ... existing types
    CONDITION_INPUT = "condition_input"
    LAMBDA_OUTPUT = "lambda_output"
    CONTROL_FLOW = "control_flow"

# Add new node type for conditional steps
class NodeType(Enum):
    # ... existing types
    CONTROL = "control"

# Conditional step specifications
PROPERTY_CONDITIONAL_SPEC = StepSpecification(
    step_type="PropertyConditional",
    node_type=NodeType.CONTROL,
    dependencies=[
        DependencySpec(
            logical_name="condition_input",
            dependency_type=DependencyType.CONDITION_INPUT,
            required=True,
            compatible_sources=["TrainingStep", "ProcessingStep", "ModelMetricsComputation"],
            semantic_keywords=["metric", "accuracy", "loss", "score", "evaluation", "performance"],
            data_type="Float",
            description="Metric value to evaluate for condition"
        )
    ],
    outputs=[
        # Conditional steps don't produce traditional outputs
        # They control execution flow instead
    ]
)

MULTI_CONDITIONAL_SPEC = StepSpecification(
    step_type="MultiConditional",
    node_type=NodeType.CONTROL,
    dependencies=[
        DependencySpec(
            logical_name="primary_metric",
            dependency_type=DependencyType.CONDITION_INPUT,
            required=True,
            compatible_sources=["TrainingStep", "ProcessingStep"],
            semantic_keywords=["accuracy", "f1", "precision", "recall"],
            data_type="Float",
            description="Primary metric for condition evaluation"
        ),
        DependencySpec(
            logical_name="secondary_metric",
            dependency_type=DependencyType.CONDITION_INPUT,
            required=False,
            compatible_sources=["TrainingStep", "ProcessingStep"],
            semantic_keywords=["loss", "error", "auc", "roc"],
            data_type="Float",
            description="Secondary metric for condition evaluation"
        )
    ],
    outputs=[]
)

FUNCTION_CONDITIONAL_SPEC = StepSpecification(
    step_type="FunctionConditional",
    node_type=NodeType.CONTROL,
    dependencies=[
        DependencySpec(
            logical_name="lambda_output",
            dependency_type=DependencyType.LAMBDA_OUTPUT,
            required=True,
            compatible_sources=["LambdaStep", "HyperparameterPrep"],
            semantic_keywords=["result", "output", "decision", "status"],
            data_type="String",
            description="Lambda function output for condition evaluation"
        )
    ],
    outputs=[]
)
```

### 2. Configuration Classes

Configuration classes for different conditional step types:

```python
class ConditionalStepConfig(BasePipelineConfig):
    """Base configuration for conditional steps."""
    condition_type: str  # 'property', 'function', 'multi'
    if_step_names: List[str] = []
    else_step_names: List[str] = []

class PropertyConditionalStepConfig(ConditionalStepConfig):
    """Configuration for property-based conditional steps."""
    condition_type: str = "property"
    property_path: str
    threshold_value: float
    comparison_operator: str = "greater_than_or_equal"  # 'greater_than', 'less_than', etc.

class MultiConditionalStepConfig(ConditionalStepConfig):
    """Configuration for multi-condition conditional steps."""
    condition_type: str = "multi"
    conditions: List[Dict[str, Any]] = []
    condition_logic: str = "AND"  # 'AND', 'OR'

class FunctionConditionalStepConfig(ConditionalStepConfig):
    """Configuration for function-based conditional steps."""
    condition_type: str = "function"
    function_output_value: str = "SUCCESS"
    comparison_operator: str = "equals"
```

### 3. Contract Definitions

Conditional steps typically don't have script contracts since they don't execute scripts:

```python
class ConditionalStepContract(ContractBase):
    """
    Contract for conditional steps.
    
    Conditional steps don't execute scripts but may need to validate
    step references and condition inputs.
    """
    expected_condition_inputs: Dict[str, str] = {}
    expected_step_references: List[str] = []
    
    def validate_step_references(self, available_steps: List[str]) -> bool:
        """Validate that referenced steps are available."""
        missing_steps = [
            step for step in self.expected_step_references 
            if step not in available_steps
        ]
        return len(missing_steps) == 0
```

## Dependency Resolution Enhancements

### 1. Semantic Matcher Updates

The SemanticMatcher needs to understand conditional step semantics:

```python
class SemanticMatcher:
    def __init__(self) -> None:
        # ... existing initialization
        
        # Add conditional step synonyms
        self.synonyms.update({
            "condition": ["condition", "check", "evaluate", "test", "validate"],
            "metric": ["metric", "score", "accuracy", "performance", "evaluation"],
            "threshold": ["threshold", "limit", "boundary", "cutoff"],
            "control": ["control", "flow", "branch", "decision", "gate"]
        })
        
        # Add conditional step abbreviations
        self.abbreviations.update({
            "cond": "condition",
            "eval": "evaluation",
            "thresh": "threshold"
        })
```

### 2. Dependency Resolver Updates

The UnifiedDependencyResolver needs to handle conditional step patterns:

```python
class UnifiedDependencyResolver:
    def _are_types_compatible(self, dep_type: DependencyType, output_type: DependencyType) -> bool:
        """Enhanced compatibility matrix including conditional types."""
        compatibility_matrix = {
            # ... existing compatibility
            DependencyType.CONDITION_INPUT: [
                DependencyType.CUSTOM_PROPERTY,
                DependencyType.PROCESSING_OUTPUT,
                DependencyType.MODEL_ARTIFACTS  # For model metrics
            ],
            DependencyType.LAMBDA_OUTPUT: [
                DependencyType.CUSTOM_PROPERTY,
                DependencyType.LAMBDA_OUTPUT
            ],
            DependencyType.CONTROL_FLOW: [
                DependencyType.CONTROL_FLOW
            ]
        }
        
        compatible_types = compatibility_matrix.get(dep_type, [])
        return output_type in compatible_types
    
    def _calculate_compatibility(self, dep_spec: DependencySpec, output_spec: OutputSpec, 
                               provider_spec: StepSpecification) -> float:
        """Enhanced compatibility calculation for conditional steps."""
        score = 0.0
        
        # ... existing compatibility logic
        
        # Special handling for conditional step dependencies
        if dep_spec.dependency_type == DependencyType.CONDITION_INPUT:
            # Boost score for metric-producing steps
            if any(keyword in output_spec.description.lower() 
                   for keyword in ["metric", "score", "accuracy", "loss"]):
                score += 0.1
                
            # Boost score for evaluation/training steps
            if provider_spec.step_type in ["TrainingStep", "ModelMetricsComputation", "XGBoostModelEval"]:
                score += 0.1
        
        return min(score, 1.0)
```

## Registry Integration

### 1. Step Names Registry Updates

Add conditional steps to the step registry:

```python
# In step_names_original.py, add conditional step entries
STEP_NAMES.update({
    # Conditional Steps
    "PropertyConditional": {
        "config_class": "PropertyConditionalStepConfig",
        "builder_step_name": "PropertyConditionalStepBuilder",
        "spec_type": "PropertyConditional",
        "sagemaker_step_type": "Condition",
        "description": "Property-based conditional step for pipeline branching",
    },
    "MultiConditional": {
        "config_class": "MultiConditionalStepConfig",
        "builder_step_name": "MultiConditionalStepBuilder",
        "spec_type": "MultiConditional",
        "sagemaker_step_type": "Condition",
        "description": "Multi-condition conditional step with logical operators",
    },
    "FunctionConditional": {
        "config_class": "FunctionConditionalStepConfig",
        "builder_step_name": "FunctionConditionalStepBuilder",
        "spec_type": "FunctionConditional",
        "sagemaker_step_type": "Condition",
        "description": "Function-based conditional step using Lambda outputs",
    },
})
```

### 2. SageMaker Step Type Validation Updates

Update the validation function to include conditional steps:

```python
def validate_sagemaker_step_type(sagemaker_type: str) -> bool:
    """Enhanced validation including conditional steps."""
    valid_types = {
        "Processing",
        "Training",
        "Transform",
        "CreateModel",
        "RegisterModel",
        "Base",
        "Utility",
        "Lambda",
        "CradleDataLoading",
        "MimsModelRegistrationProcessing",
        "Condition",  # New type for conditional steps
    }
    return sagemaker_type in valid_types
```

## Pipeline DAG Integration

### 1. DAG Extensions for Conditional Steps

Conditional steps require minimal extensions to the existing PipelineDAG:

```python
class PipelineDAG:
    def __init__(self, nodes: Optional[List[str]] = None, edges: Optional[List[tuple]] = None):
        # ... existing initialization
        self.conditional_metadata: Dict[str, Dict[str, List[str]]] = {}
    
    def add_conditional_step(self, step_name: str, if_steps: List[str], else_steps: List[str]) -> None:
        """
        Add conditional step metadata without creating dependency edges.
        
        Args:
            step_name: Name of the conditional step
            if_steps: List of step names to execute if condition is true
            else_steps: List of step names to execute if condition is false
        """
        # Ensure the conditional step node exists
        if step_name not in self.nodes:
            self.add_node(step_name)
        
        # Store conditional metadata
        self.conditional_metadata[step_name] = {
            'if_steps': if_steps,
            'else_steps': else_steps
        }
        
        logger.info(f"Added conditional step '{step_name}' with if_steps={if_steps}, else_steps={else_steps}")
    
    def is_conditional_step(self, step_name: str) -> bool:
        """Check if a step is a conditional step."""
        return step_name in self.conditional_metadata
    
    def get_conditional_branches(self, step_name: str) -> Dict[str, List[str]]:
        """Get if/else branches for a conditional step."""
        return self.conditional_metadata.get(step_name, {'if_steps': [], 'else_steps': []})
    
    def get_all_conditional_steps(self) -> List[str]:
        """Get all conditional step names."""
        return list(self.conditional_metadata.keys())
```

### 2. Natural Dependency Structure

Conditional steps work with natural data dependencies without requiring special edges:

```python
# Example: Model training with conditional registration/retraining
dag = PipelineDAG()

# Add all nodes
dag.add_node("training")
dag.add_node("model_registration")  # if branch
dag.add_node("retraining")         # else branch
dag.add_node("accuracy_check")     # conditional step

# Natural data dependencies (these ensure correct topological ordering):
dag.add_edge("training", "model_registration")  # Registration needs the trained model
dag.add_edge("training", "retraining")          # Retraining needs training output/data
dag.add_edge("training", "accuracy_check")      # Condition evaluation needs training metrics

# Conditional metadata (no additional edges needed):
dag.add_conditional_step("accuracy_check", 
                        if_steps=["model_registration"], 
                        else_steps=["retraining"])
```

**Topological Order Result**: `["training", "model_registration", "retraining", "accuracy_check"]`

This ensures:
1. **Training runs first** - produces model and metrics
2. **Branch steps are instantiated** - both branches are created with proper inputs
3. **Conditional step is created last** - can reference already-instantiated branch step instances
4. **SageMaker handles runtime branching** - only selected branch executes at runtime

### 3. Pipeline Assembler Integration

Minimal changes to PipelineAssembler to handle conditional step instantiation:

```python
class PipelineAssembler:
    def _instantiate_step(self, step_name: str) -> Step:
        """
        Enhanced step instantiation with conditional step support.
        
        The existing topological ordering from dag.topological_sort() already ensures
        that branch steps are created before conditional steps due to natural data dependencies.
        """
        builder = self.step_builders[step_name]

        # Get dependency steps (existing logic)
        dependencies = []
        for dep_name in self.dag.get_dependencies(step_name):
            if dep_name in self.step_instances:
                dependencies.append(self.step_instances[dep_name])

        # Extract inputs from dependencies (existing logic)
        inputs = {}
        if step_name in self.step_messages:
            for input_name, message in self.step_messages[step_name].items():
                # ... existing input extraction logic ...
                pass

        # Generate outputs (existing logic)
        outputs = self._generate_outputs(step_name)

        # Create step kwargs
        kwargs = {
            "inputs": inputs,
            "outputs": outputs,
            "dependencies": dependencies,
            "enable_caching": True,
        }

        # NEW: Handle conditional steps
        if self.dag.is_conditional_step(step_name):
            branches = self.dag.get_conditional_branches(step_name)
            
            # Get already-instantiated step instances for branches
            if_steps = []
            for if_step_name in branches['if_steps']:
                if if_step_name in self.step_instances:
                    if_steps.append(self.step_instances[if_step_name])
                else:
                    logger.warning(f"If-step '{if_step_name}' not found in step_instances")
            
            else_steps = []
            for else_step_name in branches['else_steps']:
                if else_step_name in self.step_instances:
                    else_steps.append(self.step_instances[else_step_name])
                else:
                    logger.warning(f"Else-step '{else_step_name}' not found in step_instances")
            
            # Add conditional step parameters
            kwargs.update({
                'if_steps': if_steps,
                'else_steps': else_steps
            })
            
            logger.info(f"Creating conditional step '{step_name}' with {len(if_steps)} if-steps and {len(else_steps)} else-steps")

        try:
            step = builder.create_step(**kwargs)
            logger.info(f"Built step {step_name}")
            return step
        except Exception as e:
            logger.error(f"Error building step {step_name}: {e}")
            raise ValueError(f"Failed to build step {step_name}: {e}") from e
```

### 4. Key Architectural Insights

1. **No Special Topological Ordering Required**: Natural data dependencies ensure branch steps are created before conditional steps
2. **Minimal Framework Changes**: Only add conditional metadata storage and branch step reference handling
3. **Backward Compatibility**: All existing functionality remains unchanged
4. **SageMaker Integration**: Conditional steps receive actual step instances as required by SageMaker's ConditionStep constructor

## Usage Examples

### 1. Property-Based Conditional Pipeline

```python
# Create DAG with conditional branching using natural dependencies
dag = PipelineDAG()

# Add all nodes
dag.add_node("training")
dag.add_node("model_registration")  # if branch
dag.add_node("retraining")         # else branch
dag.add_node("accuracy_check")     # conditional step

# Natural data dependencies (ensure correct topological ordering):
dag.add_edge("training", "model_registration")  # Registration needs the trained model
dag.add_edge("training", "retraining")          # Retraining needs training output/data
dag.add_edge("training", "accuracy_check")      # Condition evaluation needs training metrics

# Conditional metadata (no additional edges needed):
dag.add_conditional_step("accuracy_check", 
                        if_steps=["model_registration"], 
                        else_steps=["retraining"])

# Configuration
config_map = {
    "training": XGBoostTrainingConfig(...),
    "accuracy_check": PropertyConditionalStepConfig(
        condition_type="property",
        property_path="properties.FinalMetricDataList[0].MetricStat.Statistic",
        threshold_value=0.85,
        comparison_operator="greater_than_or_equal",
        if_step_names=["model_registration"],
        else_step_names=["retraining"]
    ),
    "model_registration": RegistrationConfig(...),
    "retraining": XGBoostTrainingConfig(...)
}

# Topological order will be: ["training", "model_registration", "retraining", "accuracy_check"]
```

### 2. Multi-Condition Pipeline

```python
# Multi-condition configuration
multi_condition_config = MultiConditionalStepConfig(
    condition_type="multi",
    conditions=[
        {
            'input_name': 'accuracy_metric',
            'comparison_operator': 'greater_than_or_equal',
            'threshold_value': 0.85
        },
        {
            'input_name': 'loss_metric',
            'comparison_operator': 'less_than',
            'threshold_value': 0.1
        }
    ],
    condition_logic="AND",
    if_step_names=["model_registration"],
    else_step_names=["hyperparameter_tuning"]
)
```

## Testing Implications

Conditional step builders should be tested for:

1. **Condition Creation**: Correct SageMaker condition object creation
2. **Input Resolution**: Proper dependency resolution for condition inputs
3. **Step Reference Handling**: Correct if/else step reference resolution
4. **Configuration Validation**: Comprehensive validation of conditional configurations
5. **Specification Compliance**: Adherence to conditional step specifications
6. **DAG Integration**: Proper integration with PipelineDAG conditional nodes
7. **Multi-Condition Logic**: Correct logical operator application
8. **Error Handling**: Proper handling of missing inputs and invalid configurations

### Recommended Test Categories

#### Property-Based Conditional Steps
- Threshold comparison validation
- Property path resolution testing
- Operator type validation

#### Multi-Condition Steps
- Multiple input resolution
- Logical operator application (AND/OR)
- Partial condition failure handling

#### Function-Based Steps
- Lambda output resolution
- Function result comparison
- Custom business logic validation

#### Integration Testing
- End-to-end pipeline execution with branching
- Dependency resolution across conditional boundaries
- DAG topology validation with conditional nodes

## Best Practices

1. **Specification-Driven Design**: Use specifications for condition input definitions
2. **Clear Configuration**: Provide clear, validated configuration options
3. **Dependency Resolution**: Leverage existing dependency resolution for condition inputs
4. **Error Handling**: Provide comprehensive error messages for configuration issues
5. **DAG Integration**: Properly integrate with PipelineDAG for step reference resolution
6. **Testing**: Comprehensive testing of conditional logic and edge cases
7. **Documentation**: Clear documentation of condition types and usage patterns

## References

### Related Design Documents
- **[Specification-Driven Design](./specification_driven_design.md)** - Core specification-driven architecture
- **[Processing Step Builder Patterns](./processing_step_builder_patterns.md)** - Processing step implementation patterns
- **[Training Step Builder Patterns](./training_step_builder_patterns.md)** - Training step implementation patterns
- **[Step Builder Patterns Summary](./step_builder_patterns_summary.md)** - Comprehensive pattern analysis

### Framework Components
- **[Dependency Resolver](../core/deps/dependency_resolver.py)** - Core dependency resolution logic
- **[Semantic Matcher](../core/deps/semantic_matcher.py)** - Semantic similarity matching
- **[Pipeline Assembler](../core/assembler/pipeline_assembler.py)** - Pipeline assembly orchestration
- **[Step Builder Base](../core/base/builder_base.py)** - Base step builder implementation

### Registry and Validation
- **[Step Names Registry](../registry/step_names.py)** - Central step name registry
- **[Step Names Original](../registry/step_names_original.py)** - Original step definitions
- **[Validation Utils](../registry/validation_utils.py)** - Step validation utilities

### SageMaker Documentation
- **[SageMaker Pipelines Conditions](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-condition)** - Official SageMaker condition documentation
- **[SageMaker Python SDK Conditions](https://sagemaker.readthedocs.io/en/stable/workflows/pipelines/sagemaker.workflow.pipelines.html#conditions)** - Python SDK condition reference

This comprehensive design provides a complete framework for implementing conditional steps that seamlessly integrate with the existing cursus architecture while maintaining all the benefits of specification-driven dependency resolution.

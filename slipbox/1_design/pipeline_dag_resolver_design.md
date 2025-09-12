---
tags:
  - design
  - api
  - dag_resolution
  - pipeline_execution
  - system_architecture
keywords:
  - pipeline DAG resolver
  - topological sorting
  - execution planning
  - data flow mapping
  - contract discovery
  - dependency resolution
  - NetworkX integration
topics:
  - pipeline execution
  - DAG resolution
  - data flow management
  - system architecture
language: python
date of note: 2025-08-24
---

# Pipeline DAG Resolver Design

## Overview

The `PipelineDAGResolver` is a core component of the Cursus pipeline execution system that transforms pipeline DAGs into executable plans with proper dependency resolution, topological ordering, and data flow mapping. It serves as the bridge between high-level pipeline definitions and low-level execution orchestration.

## Architecture

### Core Components

```
PipelineDAGResolver
├── PipelineExecutionPlan (Data Structure)
├── DAG Validation & Analysis
├── Topological Sorting Engine
├── Data Flow Mapping System
└── Contract Discovery System
```

### Key Classes

#### PipelineDAGResolver
- **Location**: `src/cursus/api/dag/pipeline_dag_resolver.py`
- **Purpose**: Main resolver class that orchestrates DAG analysis and execution plan creation
- **Dependencies**: NetworkX, Cursus core components, registry helpers

#### PipelineExecutionPlan
- **Purpose**: Immutable data structure representing a resolved execution plan
- **Components**:
  - `execution_order`: Topologically sorted step sequence
  - `step_configs`: Configuration for each step
  - `dependencies`: Dependency mapping between steps
  - `data_flow_map`: Input/output channel mappings

## Design Principles

### 1. Separation of Concerns
- **DAG Analysis**: Pure graph operations using NetworkX
- **Contract Discovery**: Dynamic resolution of step specifications
- **Data Flow Mapping**: Channel compatibility and path resolution
- **Execution Planning**: Immutable plan generation

### 2. Extensibility
- **Pluggable Contract Discovery**: Registry-based step resolution
- **Flexible Channel Matching**: Multiple compatibility strategies
- **Configurable Validation**: Extensible integrity checking

### 3. Robustness
- **Cycle Detection**: Prevents infinite execution loops
- **Dependency Validation**: Ensures all dependencies exist
- **Graceful Degradation**: Fallback strategies for missing contracts

## Enhanced Step Configuration Resolution

### Integration with Dynamic Template System

The PipelineDAGResolver incorporates intelligent step configuration resolution capabilities inspired by the **Dynamic Template System** design. This enhancement bridges the gap between basic DAG structure and rich step configuration management.

**Key Features**:
- **Intelligent Config Matching**: Uses the same resolution strategies as `DynamicPipelineTemplate`
- **Multiple Resolution Methods**: Direct name matching, metadata mapping, job type matching, semantic similarity
- **Backward Compatibility**: Works with basic DAGs (empty configs) and enhanced DAGs (populated configs)
- **Preview Capabilities**: Provides resolution preview for debugging and validation

### Configuration Resolution Flow

```python
# Enhanced resolver initialization with config support
resolver = PipelineDAGResolver(
    dag=my_pipeline_dag,
    config_path="./config/pipeline_config.json",  # Optional
    metadata={"config_types": {"step1": "XGBoostTraining"}}  # Optional
)

# Create execution plan with populated step configs
execution_plan = resolver.create_execution_plan()

# Step configs are now populated with actual configuration objects
for step_name, step_config in execution_plan.step_configs.items():
    print(f"{step_name}: {step_config.get('job_type', 'N/A')}")
```

### Resolution Strategies

The resolver uses a **tiered resolution approach** (from highest to lowest priority):

1. **Direct Name Matching**: Exact matches between DAG node names and config keys
2. **Metadata Mapping**: Uses `config_types` metadata for explicit node-to-config mapping
3. **Job Type + Config Type Matching**: Combines job type detection with config type inference
4. **Semantic Similarity**: Uses semantic matching for similar terms (e.g., "train" → "training")
5. **Pattern-Based Matching**: Regex patterns for common step type detection

### Integration Benefits

**For Pipeline Runtime Testing**:
- **Rich Step Configs**: Execution plans now contain actual step configurations instead of empty dicts
- **Better Data Flow**: Enhanced data flow mapping using config input/output specifications
- **Improved Validation**: Step-specific validation using configuration metadata
- **Debugging Support**: Config resolution preview helps identify mapping issues

**For Development Workflow**:
- **Seamless Integration**: Works with existing pipeline development patterns
- **Flexible Usage**: Optional config support - works with or without configurations
- **Error Handling**: Graceful fallback when configs cannot be resolved
- **Performance**: Minimal overhead when configs are not provided

## Core Functionality

### DAG Resolution Process

The `create_execution_plan()` method orchestrates the entire resolution process:

```python
def create_execution_plan(self) -> PipelineExecutionPlan:
    """Create topologically sorted execution plan with optional step config resolution."""
    # 1. Validate DAG integrity
    if not nx.is_directed_acyclic_graph(self.graph):
        raise ValueError("Pipeline contains cycles")
    
    # 2. Perform topological sorting
    execution_order = list(nx.topological_sort(self.graph))
    
    # 3. Resolve step configurations (if available)
    step_configs = self._resolve_step_configs(execution_order)
    
    # 4. Build dependency mapping
    dependencies = {
        name: list(self.graph.predecessors(name))
        for name in execution_order
    }
    
    # 5. Create data flow mappings
    data_flow_map = self._build_data_flow_map()
    
    return PipelineExecutionPlan(...)
```

**Implementation Details:**
- Uses NetworkX's `is_directed_acyclic_graph()` for cycle detection
- Leverages `topological_sort()` for dependency-aware ordering
- Integrates with `StepConfigResolver` for configuration resolution
- Handles both config-enabled and config-less DAG resolution

### Topological Sorting Implementation

The resolver converts PipelineDAG to NetworkX format for graph analysis:

```python
def _build_networkx_graph(self) -> nx.DiGraph:
    """Convert pipeline DAG to NetworkX graph."""
    graph = nx.DiGraph()
    
    # Add nodes from the DAG
    for node in self.dag.nodes:
        graph.add_node(node)
    
    # Add edges from the DAG
    for src, dst in self.dag.edges:
        graph.add_edge(src, dst)
    
    return graph
```

**Implementation Strategy:**
- Direct conversion from PipelineDAG structure to NetworkX DiGraph
- Preserves all node and edge relationships
- Enables powerful graph algorithms (cycle detection, topological sorting, etc.)
- Supports complex dependency analysis through NetworkX methods

### Contract Discovery Implementation

The `_discover_step_contract()` method implements dynamic contract discovery:

```python
def _discover_step_contract(self, step_name: str) -> Optional[ScriptContract]:
    """Dynamically discover step contract using registry helper functions."""
    try:
        # 1. Convert step name to canonical name
        canonical_name = get_canonical_name_from_file_name(step_name)
        if not canonical_name:
            return None
        
        # 2. Get specification from canonical name
        step_spec = self._get_step_specification(canonical_name)
        if not step_spec:
            return None
        
        # 3. Extract contract from specification
        if hasattr(step_spec, 'script_contract') and step_spec.script_contract:
            return step_spec.script_contract
        
        return None
    except Exception as e:
        logger.warning(f"Failed to discover contract for step {step_name}: {e}")
        return None
```

**Discovery Strategy:**
1. **Name Resolution**: Uses registry helper `get_canonical_name_from_file_name()`
2. **Specification Lookup**: Calls `_get_step_specification()` for dynamic import
3. **Contract Extraction**: Accesses `script_contract` attribute from specification
4. **Error Handling**: Graceful fallback on any failure in the chain

### Specification Resolution Implementation

The `_get_step_specification()` method handles dynamic specification loading:

```python
def _get_step_specification(self, canonical_name: str) -> Optional[StepSpecification]:
    """Get step specification from canonical name using dynamic import."""
    try:
        # 1. Get spec type from canonical name
        spec_type = get_spec_step_type(canonical_name)
        if not spec_type:
            return None
        
        # 2. Build module path using naming convention
        module_name = self._spec_type_to_module_name(spec_type)
        module_path = f"cursus.steps.specs.{module_name}"
        
        # 3. Dynamic import
        spec_module = importlib.import_module(module_path)
        
        # 4. Get specification instance using multiple strategies
        # Strategy A: Look for getter function
        spec_getter_name = f"get_{module_name}"
        if hasattr(spec_module, spec_getter_name):
            return getattr(spec_module, spec_getter_name)()
        
        # Strategy B: Look for direct spec class
        if hasattr(spec_module, spec_type):
            spec_class = getattr(spec_module, spec_type)
            return spec_class()
        
        # Strategy C: Look for common variable names
        for var_name in ['SPEC', 'spec', f'{canonical_name.upper()}_SPEC']:
            if hasattr(spec_module, var_name):
                return getattr(spec_module, var_name)
        
        return None
    except ImportError:
        return None
```

**Implementation Features:**
- **Naming Convention**: Converts spec types to module names (e.g., "XGBoostTrainingSpec" → "xgboost_training_spec")
- **Multiple Access Patterns**: Supports getter functions, direct classes, and variable access
- **Robust Error Handling**: Handles import failures gracefully
- **Flexible Module Structure**: Adapts to different specification module organizations

### Data Flow Mapping Implementation

The `_build_data_flow_map()` method creates contract-based channel mappings:

```python
def _build_data_flow_map(self) -> Dict[str, Dict[str, str]]:
    """Build data flow map using contract-based channel definitions."""
    data_flow = {}
    
    for step_name in self.graph.nodes():
        inputs = {}
        
        # Get step contract dynamically
        step_contract = self._discover_step_contract(step_name)
        if not step_contract:
            # Fallback to generic approach
            for i, dep_step in enumerate(self.graph.predecessors(step_name)):
                inputs[f"input_{i}"] = f"{dep_step}:output"
            data_flow[step_name] = inputs
            continue
        
        # Map each expected input channel to dependency outputs
        for input_channel, input_path in step_contract.expected_input_paths.items():
            for dep_step in self.graph.predecessors(step_name):
                dep_contract = self._discover_step_contract(dep_step)
                if dep_contract:
                    # Find compatible output channel
                    compatible_output = self._find_compatible_output(
                        input_channel, input_path,
                        dep_contract.expected_output_paths
                    )
                    if compatible_output:
                        inputs[input_channel] = f"{dep_step}:{compatible_output}"
                        break
        
        data_flow[step_name] = inputs
    
    return data_flow
```

**Implementation Strategy:**
- **Contract-First Approach**: Uses discovered contracts to define precise channel mappings
- **Graceful Fallback**: Falls back to generic `input_N:output` when contracts unavailable
- **Dependency Analysis**: Iterates through graph predecessors for each step
- **Channel Compatibility**: Uses `_find_compatible_output()` for intelligent matching

### Channel Compatibility Implementation

The `_find_compatible_output()` method implements multi-strategy channel matching:

```python
def _find_compatible_output(self, input_channel: str, input_path: str, 
                           output_channels: Dict[str, str]) -> Optional[str]:
    """Find compatible output channel for given input requirements."""
    
    # Strategy 1: Direct channel name matching
    if input_channel in output_channels:
        return input_channel
    
    # Strategy 2: Path-based compatibility
    for output_channel, output_path in output_channels.items():
        if self._are_paths_compatible(input_path, output_path):
            return output_channel
    
    # Strategy 3: Semantic matching for common patterns
    semantic_matches = {
        'input_path': ['output_path', 'model_path', 'data_path'],
        'model_path': ['model_output_path', 'output_path'],
        'data_path': ['output_path', 'processed_data_path'],
        'hyperparameters_s3_uri': ['config_path', 'hyperparameters_path']
    }
    
    if input_channel in semantic_matches:
        for candidate in semantic_matches[input_channel]:
            if candidate in output_channels:
                return candidate
    
    # Strategy 4: Fallback to first available output
    if output_channels:
        return next(iter(output_channels.keys()))
    
    return None
```

**Matching Strategies:**
1. **Direct Matching**: Exact channel name correspondence
2. **Path Compatibility**: SageMaker path convention analysis
3. **Semantic Matching**: Predefined logical relationships between channels
4. **Fallback Strategy**: First available output when no better match exists

### Path Compatibility Implementation

The `_are_paths_compatible()` method implements SageMaker-aware path matching:

```python
def _are_paths_compatible(self, input_path: str, output_path: str) -> bool:
    """Check if input and output paths are compatible based on SageMaker conventions."""
    
    # SageMaker path compatibility rules
    compatible_mappings = [
        ('/opt/ml/model', '/opt/ml/model'),  # Model artifacts
        ('/opt/ml/input/data', '/opt/ml/output/data'),  # Data flow
        ('/opt/ml/output', '/opt/ml/input/data'),  # Output to input
    ]
    
    for input_pattern, output_pattern in compatible_mappings:
        if input_pattern in input_path and output_pattern in output_path:
            return True
    
    # Generic compatibility: same base directory structure
    input_parts = Path(input_path).parts
    output_parts = Path(output_path).parts
    
    if len(input_parts) >= 2 and len(output_parts) >= 2:
        if input_parts[-2:] == output_parts[-2:]:
            return True
    
    return False
```

**Compatibility Rules:**
- **SageMaker Conventions**: Recognizes standard ML platform path patterns
- **Directory Structure**: Matches based on common directory hierarchies
- **Flexible Matching**: Supports both exact and pattern-based compatibility

### Validation Implementation

The `validate_dag_integrity()` method performs comprehensive DAG validation:

```python
def validate_dag_integrity(self) -> Dict[str, List[str]]:
    """Validate DAG integrity and return issues if found."""
    issues = {}
    
    # Check for cycles
    try:
        list(nx.topological_sort(self.graph))
    except nx.NetworkXUnfeasible:
        cycles = list(nx.simple_cycles(self.graph))
        issues["cycles"] = [f"Cycle detected: {' -> '.join(cycle)}" for cycle in cycles]
    
    # Check for dangling dependencies
    for src, dst in self.dag.edges:
        if src not in self.dag.nodes:
            if "dangling_dependencies" not in issues:
                issues["dangling_dependencies"] = []
            issues["dangling_dependencies"].append(
                f"Edge references non-existent source node: {src}"
            )
        if dst not in self.dag.nodes:
            if "dangling_dependencies" not in issues:
                issues["dangling_dependencies"] = []
            issues["dangling_dependencies"].append(
                f"Edge references non-existent destination node: {dst}"
            )
    
    # Check for isolated nodes
    isolated_nodes = []
    for node in self.dag.nodes:
        if self.graph.degree(node) == 0:
            isolated_nodes.append(node)
    
    if isolated_nodes:
        issues["isolated_nodes"] = [f"Node has no connections: {node}" for node in isolated_nodes]
    
    return issues
```

**Validation Categories:**
1. **Cycle Detection**: Uses NetworkX's topological sort failure to detect cycles
2. **Dangling Dependencies**: Validates that all edge endpoints exist as nodes
3. **Isolated Nodes**: Identifies nodes with no incoming or outgoing connections
4. **Structured Reporting**: Returns categorized issues for targeted remediation

**Implementation Features:**
- **Non-Destructive**: Validation doesn't modify the DAG structure
- **Comprehensive Coverage**: Checks multiple types of structural issues
- **Actionable Results**: Provides specific error messages for each issue type
- **Graceful Handling**: Continues validation even when individual checks fail

## Related Design Documents

### Core Data Structures

The PipelineDAGResolver works with several key data structures that have their own design documentation:

- **[Pipeline DAG](pipeline_dag.md)**: Core DAG structure and dependency management that provides the mathematical framework for pipeline topology and execution ordering
- **[Step Specification](step_specification.md)**: Step specification format and validation system that defines the contract interface for dynamic discovery
- **[Script Contract](script_contract.md)**: Script contract specifications that define execution interfaces and provide the foundation for data flow mapping
- **[Config Base](config.md)**: Base configuration classes and validation patterns used in step configuration resolution

### System Integration Documents

- **[Pipeline Runtime Execution Layer Design](pipeline_runtime_execution_layer_design.md)**: High-level pipeline orchestration layer that uses PipelineDAGResolver for execution planning
- **[Pipeline Runtime Core Engine Design](pipeline_runtime_core_engine_design_OUTDATED.md)**: ⚠️ **OUTDATED** - Core execution engine components that work with resolved execution plans
- **[Pipeline Runtime System Integration Design](pipeline_runtime_system_integration_design_OUTDATED.md)**: ⚠️ **OUTDATED** - Integration with existing Cursus components including DAG analysis and dependency resolution
- **[Dependency Resolver](dependency_resolver.md)**: Dependency resolution system that complements DAG resolution with intelligent matching

### Registry and Builder Integration

- **[Step Builder](step_builder.md)**: Builder pattern used for step instantiation and registry integration
- **[Step Config Resolver](step_config_resolver.md)**: Configuration resolution system that maps DAG nodes to step configurations

## Integration Points

### Registry System Integration

The resolver integrates with Cursus registry components:

```python
# Registry helper functions
from cursus.steps.step_builder_registry import (
    get_canonical_name_from_file_name,
    get_spec_step_type
)
```

### Core Component Integration

```python
# Core Cursus components
from cursus.core.base.config_base import BasePipelineConfig
from cursus.core.base.contract_base import ScriptContract
from cursus.core.base.specification_base import StepSpecification
```

### API Layer Integration

```python
# API layer components
from . import PipelineDAG  # Relative import from api.dag
```

## Usage Patterns

### Basic Usage

```python
from cursus.api.dag import PipelineDAG
from cursus.api.dag.pipeline_dag_resolver import PipelineDAGResolver

# Create DAG
dag = PipelineDAG(
    nodes=["preprocessing", "training", "evaluation"],
    edges=[("preprocessing", "training"), ("training", "evaluation")]
)

# Resolve execution plan
resolver = PipelineDAGResolver(dag)
plan = resolver.create_execution_plan()

# Access execution details
print(f"Execution order: {plan.execution_order}")
print(f"Data flow: {plan.data_flow_map}")
```

### Advanced Usage with Validation

```python
# Validate DAG before resolution
issues = resolver.validate_dag_integrity()
if issues:
    print(f"DAG validation issues: {issues}")
else:
    plan = resolver.create_execution_plan()
    # Proceed with execution
```

## Error Handling

### Contract Discovery Failures
- **Missing Registry Entry**: Graceful fallback to generic mapping
- **Import Errors**: Log warning and continue with basic resolution
- **Contract Validation**: Skip invalid contracts, use fallback

### DAG Validation Failures
- **Cycles**: Raise `ValueError` with cycle information
- **Missing Dependencies**: Include in validation report
- **Isolated Nodes**: Report as warnings, not errors

## Performance Considerations

### Optimization Strategies
1. **Lazy Contract Discovery**: Only discover contracts when needed
2. **Caching**: Cache resolved specifications for repeated use
3. **Minimal Graph Operations**: Efficient NetworkX usage
4. **Early Validation**: Fail fast on invalid DAGs

### Scalability
- **Linear Complexity**: O(V + E) for topological sorting
- **Bounded Discovery**: Contract discovery limited by step count
- **Memory Efficient**: Immutable plan structures

## Testing Strategy

### Unit Tests
- **DAG Resolution**: Test topological sorting correctness
- **Contract Discovery**: Mock registry functions for isolation
- **Data Flow Mapping**: Verify channel compatibility logic
- **Validation**: Test all validation scenarios

### Integration Tests
- **End-to-End**: Full pipeline resolution with real specifications
- **Registry Integration**: Test with actual registry components
- **Error Scenarios**: Validate error handling paths

## Future Enhancements

### Planned Improvements
1. **Parallel Execution Planning**: Support for concurrent step execution
2. **Resource Optimization**: Memory and compute resource planning
3. **Dynamic Replanning**: Runtime plan modification capabilities
4. **Advanced Validation**: Semantic contract validation

### Extension Points
1. **Custom Channel Matchers**: Pluggable compatibility strategies
2. **Alternative Sorting**: Support for different ordering algorithms
3. **Contract Providers**: Multiple contract discovery mechanisms
4. **Validation Rules**: Extensible validation framework

## Migration Notes

### From Runtime Execution Module
The `PipelineDAGResolver` was moved from `cursus.validation.runtime.execution` to `cursus.api.dag` to reflect its general-purpose nature beyond just runtime testing.

#### Import Path Changes
```python
# Old import
from cursus.validation.runtime.execution.pipeline_dag_resolver import PipelineDAGResolver

# New import
from cursus.api.dag.pipeline_dag_resolver import PipelineDAGResolver
```

#### Affected Components
- Test files moved to `test/api/dag/`
- Import references updated in validation runtime modules
- Project planning documents updated with new paths

## Conclusion

The `PipelineDAGResolver` provides a robust, extensible foundation for pipeline execution planning. Its integration with the Cursus registry system enables dynamic contract discovery while maintaining fallback strategies for reliability. The clean separation of concerns and comprehensive validation make it suitable for both development and production use cases.

The resolver's design supports the broader Cursus architecture goals of modularity, testability, and maintainability while providing the performance characteristics needed for complex pipeline orchestration.

---
tags:
  - archive
  - design
  - pipeline_runtime_testing
  - simplified_architecture
  - validation_framework
  - node_to_script_resolution
keywords:
  - pipeline runtime testing
  - script functionality validation
  - data transfer consistency
  - node-to-script resolution
  - intelligent file discovery
  - dual identity management
topics:
  - pipeline runtime testing
  - script validation
  - data flow testing
  - simplified architecture
language: python
date of note: 2025-09-09
---

# Pipeline Runtime Testing Simplified Design

## Overview

This document outlines the core design for the Pipeline Runtime Testing system, which validates script functionality and data transfer consistency for pipeline development with intelligent node-to-script resolution. The system addresses the fundamental challenge of associating PipelineDAG node names with corresponding ScriptExecutionSpec objects.

## Core Architecture Components

The runtime testing system consists of four main components that work together to provide comprehensive pipeline validation:

1. **[PipelineTestingSpecBuilder](pipeline_testing_spec_builder_design.md)** - Builds PipelineTestingSpec from PipelineDAG with intelligent node-to-script resolution
2. **[RuntimeTester](runtime_tester_design.md)** - Executes script testing and validation with enhanced data compatibility
3. **[ScriptExecutionSpec](script_execution_spec_design.md)** - Defines script execution parameters and path specifications with dual identity management
4. **[PipelineTestingSpec](pipeline_testing_spec_design.md)** - Contains complete pipeline testing configuration and orchestration

## Node-to-Script Resolution Challenge

When traversing a PipelineDAG in pipeline_spec, the system needs to associate DAG node names (like `"TabularPreprocessing_training"`) with corresponding ScriptExecutionSpec objects. The challenge is that:

- **DAG node names** are canonical names with job type suffixes (e.g., `"TabularPreprocessing_training"`)
- **ScriptExecutionSpec** has dual identity: `script_name` (file identity) and `step_name` (DAG node identity)
- **Script files** use snake_case naming (e.g., `"tabular_preprocessing.py"`)

## Solution: Registry-Based Resolution with File Verification

The **[PipelineTestingSpecBuilder](pipeline_testing_spec_builder_design.md)** provides comprehensive node-to-script resolution through a multi-step process:

### Resolution Process

```python
def resolve_script_execution_spec_from_node(self, node_name: str) -> ScriptExecutionSpec:
    """
    Multi-step resolution process:
    1. Registry-based canonical name extraction
    2. PascalCase to snake_case conversion with special cases
    3. Workspace-first file discovery with fuzzy matching
    4. ScriptExecutionSpec creation with dual identity
    """
    from cursus.registry.step_names import get_step_name_from_spec_type
    
    # Step 1: Get canonical step name using existing registry function
    canonical_name = get_step_name_from_spec_type(node_name)
    
    # Step 2: Convert to script name with special case handling
    script_name = self._canonical_to_script_name(canonical_name)
    
    # Step 3: Find actual script file with verification
    script_path = self._find_script_file(script_name)
    
    # Step 4: Create ScriptExecutionSpec with dual identity
    return ScriptExecutionSpec(
        script_name=script_name,      # For file discovery (snake_case)
        step_name=node_name,          # For DAG node matching (PascalCase + job type)
        script_path=str(script_path),
        # ... additional configuration
    )
```

### Key Resolution Examples

**Standard Processing Step**:
```
"TabularPreprocessing_training" → "TabularPreprocessing" → "tabular_preprocessing" → "tabular_preprocessing.py"
```

**Complex Technical Term**:
```
"XGBoostModelEval_evaluation" → "XGBoostModelEval" → "xgboost_model_eval" → "xgboost_model_evaluation.py" (via fuzzy matching)
```

### Directory Structure

```
test_data_dir/
├── scripts/                           # Test workspace scripts (priority 1)
│   ├── tabular_preprocessing.py
│   ├── xgboost_training.py
│   ├── xgboost_model_evaluation.py
│   └── ...
├── .specs/                            # ScriptExecutionSpec storage (hidden)
│   ├── tabular_preprocessing_runtime_test_spec.json
│   ├── xgboost_training_runtime_test_spec.json
│   └── ...
├── input/                             # Test input data
├── output/                            # Test output data
└── results/                           # Test execution results
```

## Enhanced Features

### Universal File Format Support
- **Format Agnostic**: Supports any file format scripts produce (CSV, JSON, Parquet, PKL, BST, ONNX, TAR.GZ, etc.)
- **Intelligent Filtering**: Smart temporary file detection excludes system files while preserving valid outputs
- **Future Proof**: Automatic support for new formats without code changes

### Logical Name Matching System
- **5-Level Matching Priority**: Exact logical → alias combinations → semantic similarity
- **Semantic Integration**: Leverages existing `SemanticMatcher` infrastructure
- **Alias Support**: Enhanced path specifications with alternative name support
- **Comprehensive Reporting**: Detailed matching results with confidence scoring

### Topological Execution
- **Dependency-Aware Testing**: Proper execution order following DAG topology
- **Data Flow Validation**: Tests actual data transfer between connected scripts
- **Comprehensive Coverage**: Ensures all DAG edges are validated
- **Early Failure Detection**: Stops execution chain when dependencies fail

## Integration with Component Designs

### PipelineTestingSpecBuilder Integration
The builder creates complete **[PipelineTestingSpec](pipeline_testing_spec_design.md)** objects:

```python
def build_from_dag(self, dag: PipelineDAG, pipeline_name: str) -> PipelineTestingSpec:
    """Build complete pipeline testing specification from DAG."""
    
    # Resolve all script specs with intelligent node-to-script resolution
    script_specs = {}
    for node_name in dag.nodes:
        script_specs[node_name] = self.resolve_script_execution_spec_from_node(node_name)
    
    # Create pipeline spec with validation
    pipeline_spec = PipelineTestingSpec(
        dag=dag,
        script_specs=script_specs,
        pipeline_name=pipeline_name,
        test_data_dir=str(self.test_data_dir),
        testing_config=self._get_default_testing_config()
    )
    
    return pipeline_spec
```

### RuntimeTester Integration
The **[RuntimeTester](runtime_tester_design.md)** executes comprehensive pipeline testing:

```python
def test_pipeline_flow_with_spec(self, pipeline_spec: PipelineTestingSpec) -> Dict[str, Any]:
    """Execute complete pipeline testing with topological ordering."""
    
    # Get topological execution order
    execution_order = pipeline_spec.get_execution_order()
    
    # Test each script and data flow in dependency order
    for node_name in execution_order:
        # Test individual script functionality
        script_result = self.test_script_with_spec(script_specs[node_name], main_params)
        
        # Test data compatibility with dependent nodes
        for edge in outgoing_edges:
            compat_result = self.test_data_compatibility_with_specs(src_spec, dst_spec)
    
    return comprehensive_results
```

### ScriptExecutionSpec Integration
The **[ScriptExecutionSpec](script_execution_spec_design.md)** manages dual identity and execution context:

```python
class ScriptExecutionSpec(BaseModel):
    """Comprehensive specification for script execution in runtime testing."""
    
    # Core Identity Fields - addresses the dual identity challenge
    script_name: str = Field(..., description="Script file name (snake_case)")
    step_name: str = Field(..., description="DAG node name (PascalCase with job type)")
    script_path: str = Field(..., description="Full path to script file")
    
    # Path Specifications with logical name mapping
    input_paths: Dict[str, str] = Field(default_factory=dict)
    output_paths: Dict[str, str] = Field(default_factory=dict)
    
    # Execution Context
    environ_vars: Dict[str, str] = Field(default_factory=dict)
    job_args: Dict[str, Any] = Field(default_factory=dict)
```

## Design Benefits

### Registry Integration
✅ **Proven Foundation**: Uses existing `get_step_name_from_spec_type` for job type handling  
✅ **Special Case Handling**: Proper conversion of compound technical terms (XGBoost, PyTorch)  
✅ **Workspace Awareness**: Prioritizes test workspace while supporting core framework fallback  

### Intelligent Resolution
✅ **File Verification**: Checks actual files with fuzzy matching fallback  
✅ **Error Recovery**: Placeholder creation and comprehensive error handling  
✅ **Dual Identity Management**: Clear separation of file identity vs DAG node identity  

### Complete Integration
✅ **Comprehensive Validation**: DAG-script consistency checking and error reporting  
✅ **Seamless Integration**: Works with all four core components  
✅ **Backward Compatibility**: Maintains existing APIs while adding enhanced features  

## Usage Example

```python
# Create builder and resolve DAG to testing spec
builder = PipelineTestingSpecBuilder(test_data_dir="/test_workspace")
pipeline_spec = builder.build_from_dag(dag, "xgboost_training_pipeline")

# Execute comprehensive pipeline testing
tester = RuntimeTester(builder)
results = tester.test_pipeline_flow_with_spec(pipeline_spec)

# Analyze results
if results["pipeline_success"]:
    print(f"✅ Pipeline testing successful!")
    print(f"Execution order: {results['execution_order']}")
    print(f"Scripts tested: {len(results['script_results'])}")
    print(f"Data flows validated: {len(results['data_flow_results'])}")
else:
    print(f"❌ Pipeline testing failed!")
    for error in results["errors"]:
        print(f"  - {error}")
```

## Implementation Status

The system has been implemented in phases with comprehensive enhancements:

- ✅ **Phase 1**: Enhanced file format support with intelligent filtering
- ✅ **Phase 2**: Logical name matching system with semantic similarity
- ✅ **Phase 3**: Full integration with topological execution ordering

For detailed implementation information, see the **[Pipeline Runtime Testing Enhancement Implementation Plan](../2_project_planning/2025-09-09_pipeline_runtime_testing_enhancement_implementation_plan.md)**.

## References

### Core Component Designs
- **[PipelineTestingSpecBuilder Design](pipeline_testing_spec_builder_design.md)** - Node-to-script resolution and builder pattern implementation
- **[RuntimeTester Design](runtime_tester_design.md)** - Execution engine and testing workflow implementation  
- **[ScriptExecutionSpec Design](script_execution_spec_design.md)** - Script execution configuration and dual identity management
- **[PipelineTestingSpec Design](pipeline_testing_spec_design.md)** - Pipeline-level configuration and orchestration

### Supporting Framework
- **[Pytest Unittest Compatibility Framework Design](pytest_unittest_compatibility_framework_design.md)** - Testing framework integration patterns

### Implementation Planning
- **[Pipeline Runtime Testing Enhancement Implementation Plan](../2_project_planning/2025-09-09_pipeline_runtime_testing_enhancement_implementation_plan.md)** - Comprehensive implementation roadmap and project status

### Foundation Documents
- **[Config Driven Design](config_driven_design.md)** - Core principles for specification-driven system architecture
- **[Design Principles](design_principles.md)** - Fundamental design patterns and architectural guidelines

## Conclusion

The Pipeline Runtime Testing system provides a robust, intelligent validation framework that addresses the core challenge of node-to-script resolution while maintaining simplicity and performance. By combining registry-based resolution, intelligent file discovery, and comprehensive validation, it enables reliable pipeline testing that scales from simple script validation to complex pipeline workflows.

The system demonstrates how thoughtful design can solve complex technical challenges while preserving the user-focused principles that make it effective for daily development use.

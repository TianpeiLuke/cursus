---
tags:
  - design
  - testing
  - runtime
  - core_engine
  - execution
keywords:
  - pipeline script executor
  - script import manager
  - data flow manager
  - core execution engine
  - script execution
topics:
  - testing framework
  - core engine design
  - script execution
language: python
date of note: 2025-08-21
---

# Pipeline Runtime Testing - Core Execution Engine Design

**Date**: August 21, 2025  
**Status**: Design Phase  
**Priority**: High  
**Scope**: Core execution engine components for pipeline runtime testing

## 🎯 Overview

This document details the design of the **Core Execution Engine** for the Pipeline Runtime Testing System. The core engine is responsible for orchestrating script execution, managing dynamic imports, and handling data flow between pipeline steps.

## 📦 Core Components

### 1. PipelineScriptExecutor

**Purpose**: Main orchestrator for pipeline script execution testing

**Key Responsibilities**:
- Orchestrate script execution in topological order
- Manage test execution workflows
- Coordinate between different testing modes
- Handle error isolation and reporting

**Class Design**:
```python
class PipelineScriptExecutor:
    """Main orchestrator for pipeline script execution testing"""
    
    def __init__(self, workspace_dir: str = "./pipeline_testing"):
        self.workspace_dir = Path(workspace_dir)
        self.script_manager = ScriptImportManager()
        self.data_manager = DataFlowManager(workspace_dir)
        self.execution_history = []
        
    def test_script_isolation(self, script_name: str, 
                             data_source: str = "synthetic") -> TestResult:
        """Test single script in isolation with specified data source"""
        
    def test_pipeline_e2e(self, pipeline_dag: Dict, 
                         data_source: str = "synthetic") -> PipelineTestResult:
        """Test complete pipeline end-to-end with data flow validation"""
        
    def test_step_transition(self, upstream_step: str, 
                           downstream_step: str) -> TransitionTestResult:
        """Test data flow compatibility between two specific steps"""
        
    def execute_pipeline_with_breakpoints(self, pipeline_dag: Dict, 
                                        breakpoints: List[str] = None) -> InteractiveExecutionResult:
        """Execute pipeline with interactive breakpoints for debugging"""
```

**Core Features**:
- **Topological Execution**: Executes scripts in proper DAG order
- **Data Flow Management**: Chains outputs to inputs between steps
- **Error Isolation**: Isolates failures to specific scripts or transitions
- **Performance Monitoring**: Tracks execution time and resource usage

### 2. ScriptImportManager

**Purpose**: Handles dynamic import and execution of pipeline scripts

**Key Responsibilities**:
- Dynamically import script modules safely
- Prepare execution contexts from configurations
- Execute script main functions with proper error handling
- Discover script paths from configurations

**Class Design**:
```python
class ScriptImportManager:
    """Handles dynamic import and execution of pipeline scripts"""
    
    def __init__(self):
        self._imported_modules = {}
        self._script_cache = {}
        
    def import_script_main(self, script_path: str) -> callable:
        """Dynamically import main function from script path"""
        
    def prepare_execution_context(self, step_config: ConfigBase, 
                                 input_data_paths: Dict, 
                                 output_base_dir: str) -> ExecutionContext:
        """Prepare all parameters for script main() function"""
        
    def execute_script_main(self, main_func: callable, 
                           context: ExecutionContext) -> ExecutionResult:
        """Execute script main function with comprehensive error handling"""
        
    def discover_script_from_config(self, config: ConfigBase) -> str:
        """Discover script path from configuration using contract information"""
```

**Core Features**:
- **Dynamic Import**: Safe dynamic importing of script modules
- **Context Preparation**: Converts configs to script execution parameters
- **Error Handling**: Comprehensive exception handling and reporting
- **Script Discovery**: Automatic script path resolution from configs

### 3. DataFlowManager

**Purpose**: Manages data flow between script executions

**Key Responsibilities**:
- Map upstream outputs to downstream inputs
- Capture and validate step outputs
- Validate data compatibility between steps
- Track data lineage through pipeline execution

**Class Design**:
```python
class DataFlowManager:
    """Manages data flow between script executions"""
    
    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)
        self.data_lineage = []
        
    def setup_step_inputs(self, step_name: str, upstream_outputs: Dict, 
                         step_contract: ScriptContract) -> Dict[str, str]:
        """Map upstream outputs to current step inputs based on contract"""
        
    def capture_step_outputs(self, step_name: str, output_paths: Dict) -> Dict[str, Any]:
        """Capture and validate step outputs after execution"""
        
    def validate_data_compatibility(self, producer_output: Any, 
                                   consumer_contract: ScriptContract) -> ValidationResult:
        """Validate data format compatibility between steps"""
        
    def create_data_lineage(self, execution_history: List[ExecutionResult]) -> DataLineage:
        """Create data lineage tracking for executed pipeline"""
```

**Core Features**:
- **Path Mapping**: Maps logical paths to physical file locations
- **Data Validation**: Validates data format and schema compatibility
- **Output Capture**: Captures and analyzes script outputs
- **Lineage Tracking**: Tracks data transformations through pipeline

## 🔧 Implementation Details

### Script Import Strategy

**Dynamic Import Process**:
1. **Module Discovery**: Locate script module using contract information
2. **Safe Import**: Import module with proper error handling
3. **Function Extraction**: Extract main function from imported module
4. **Caching**: Cache imported modules for performance

**Error Handling**:
- Import errors are captured and reported with recommendations
- Missing dependencies are identified and reported
- Syntax errors in scripts are caught and analyzed

### Execution Context Preparation

**Context Components**:
```python
@dataclass
class ExecutionContext:
    input_paths: Dict[str, str]
    output_paths: Dict[str, str]
    environ_vars: Dict[str, str]
    job_args: argparse.Namespace
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for script main function call"""
        return {
            'input_paths': self.input_paths,
            'output_paths': self.output_paths,
            'environ_vars': self.environ_vars,
            'job_args': self.job_args
        }
```

**Context Preparation Process**:
1. **Path Resolution**: Resolve logical paths to physical locations
2. **Environment Setup**: Prepare environment variables from configuration
3. **Argument Preparation**: Create argparse.Namespace from configuration
4. **Validation**: Validate context completeness before execution

### Data Flow Management

**Data Flow Process**:
1. **Input Mapping**: Map upstream outputs to downstream input requirements
2. **Path Resolution**: Resolve file paths and ensure data availability
3. **Schema Validation**: Validate data schema compatibility
4. **Quality Checks**: Perform data quality validation
5. **Lineage Tracking**: Record data transformations and dependencies

## 📊 Performance Considerations

### Optimization Strategies

**Import Optimization**:
- Module caching to avoid repeated imports
- Lazy loading of heavy dependencies
- Import error recovery mechanisms

**Execution Optimization**:
- Parallel execution for independent scripts
- Resource monitoring and management
- Memory usage optimization

**Data Flow Optimization**:
- Efficient data transfer between steps
- Streaming for large datasets
- Caching of intermediate results

## 🔒 Error Handling and Recovery

### Error Categories

**Import Errors**:
- Missing script files
- Import dependency issues
- Syntax errors in scripts

**Execution Errors**:
- Runtime exceptions in scripts
- Resource exhaustion
- Data compatibility issues

**Data Flow Errors**:
- Missing input data
- Schema mismatches
- File format incompatibilities

### Recovery Strategies

**Graceful Degradation**:
- Continue execution with remaining scripts when possible
- Provide detailed error reporting for failed steps
- Offer recommendations for error resolution

**Retry Mechanisms**:
- Automatic retry for transient failures
- Configurable retry policies
- Circuit breaker patterns for persistent failures

## 📚 Integration Points

### Configuration System Integration
```python
from cursus.core.compiler.config_resolver import ConfigResolver

class ConfigurationIntegration:
    """Integration with existing configuration system"""
    
    def resolve_step_config(self, step_name: str, pipeline_dag: Dict) -> ConfigBase:
        """Use existing config resolver to get step configuration"""
        
    def extract_script_path(self, config: ConfigBase) -> str:
        """Extract script path from configuration using contract"""
```

### Contract System Integration
```python
from cursus.steps.contracts import *

class ContractIntegration:
    """Integration with existing contract system"""
    
    def get_script_contract(self, script_name: str) -> ScriptContract:
        """Get script contract for validation and setup"""
        
    def validate_input_requirements(self, contract: ScriptContract, 
                                   available_inputs: Dict) -> ValidationResult:
        """Validate that required inputs are available"""
```

## 🎯 Success Criteria

### Functional Requirements
- ✅ Successfully import and execute all pipeline scripts
- ✅ Handle script execution errors gracefully
- ✅ Validate data flow between connected scripts
- ✅ Provide comprehensive error reporting

### Performance Requirements
- ✅ Script execution time < 30 seconds per script (synthetic data)
- ✅ Memory usage < 2GB peak for typical pipelines
- ✅ Support for parallel execution of independent scripts
- ✅ Efficient data transfer between pipeline steps

### Reliability Requirements
- ✅ Robust error handling for all failure scenarios
- ✅ Graceful degradation when individual scripts fail
- ✅ Comprehensive logging and debugging information
- ✅ Recovery mechanisms for transient failures

## 📚 Cross-References

### **Master Design Document**
- **[Pipeline Runtime Testing Master Design](pipeline_runtime_testing_master_design.md)**: Master design document that provides overall system architecture

### **Related Component Designs**
- **[Data Management Layer Design](pipeline_runtime_data_management_design.md)**: Data generation, S3 integration, and compatibility validation
- **[Testing Modes Design](pipeline_runtime_testing_modes_design.md)**: Isolation, pipeline, and deep dive testing modes
- **[System Integration Design](pipeline_runtime_system_integration_design.md)**: Integration with existing Cursus components

### **Foundation Documents**
- **[Script Contract](script_contract.md)**: Script contract specifications that define execution interfaces
- **[Step Specification](step_specification.md)**: Step specification system that provides execution context
- **[Pipeline DAG](pipeline_dag.md)**: DAG structure that drives execution ordering

### **Implementation Planning**
- **[Pipeline Runtime Testing Master Implementation Plan](../2_project_planning/2025-08-21_pipeline_runtime_testing_master_implementation_plan.md)**: Implementation plan with detailed development phases

---

**Document Status**: Complete  
**Next Steps**: Review related component designs and proceed with implementation  
**Part of**: Pipeline Runtime Testing System Design

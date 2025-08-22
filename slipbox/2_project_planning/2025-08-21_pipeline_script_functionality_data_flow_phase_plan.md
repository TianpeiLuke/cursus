---
tags:
  - project
  - implementation
  - pipeline_testing
  - data_flow
  - phase_2
keywords:
  - pipeline execution
  - data flow validation
  - DAG integration
  - error handling
  - monitoring
  - cursus integration
topics:
  - pipeline testing system
  - data flow validation
  - system integration
  - implementation planning
language: python
date of note: 2025-08-21
---

# Pipeline Script Functionality Testing - Data Flow Testing Phase Implementation Plan

## Phase Overview

**Duration**: Weeks 3-4 (2 weeks)  
**Focus**: Pipeline-level testing capabilities and data flow validation  
**Dependencies**: Foundation Phase completion  
**Team Size**: 2-3 developers  

## Phase Objectives

1. Implement pipeline-level execution with dependency resolution
2. Create data flow validation and compatibility checking
3. Integrate with existing Cursus DAG and configuration systems
4. Develop comprehensive error handling and recovery mechanisms
5. Build pipeline visualization and monitoring capabilities

## Week 3: Pipeline Execution Engine

### Day 1-2: Pipeline DAG Integration
```python
# src/cursus/testing/pipeline_dag_resolver.py
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import networkx as nx
from cursus.core.dag import PipelineDAG
from cursus.steps.configs import StepConfig

@dataclass
class PipelineExecutionPlan:
    """Execution plan for pipeline with topological ordering."""
    execution_order: List[str]
    step_configs: Dict[str, StepConfig]
    dependencies: Dict[str, List[str]]
    data_flow_map: Dict[str, Dict[str, str]]

class PipelineDAGResolver:
    """Resolves pipeline DAG into executable plan."""
    
    def __init__(self, dag: PipelineDAG):
        self.dag = dag
        self.graph = self._build_networkx_graph()
    
    def _build_networkx_graph(self) -> nx.DiGraph:
        """Convert pipeline DAG to NetworkX graph."""
        graph = nx.DiGraph()
        
        for step_name, step_config in self.dag.steps.items():
            graph.add_node(step_name, config=step_config)
            
            # Add dependency edges
            for dependency in step_config.depends_on:
                graph.add_edge(dependency, step_name)
        
        return graph
    
    def create_execution_plan(self) -> PipelineExecutionPlan:
        """Create topologically sorted execution plan."""
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Pipeline contains cycles")
        
        execution_order = list(nx.topological_sort(self.graph))
        step_configs = {
            name: self.graph.nodes[name]['config'] 
            for name in execution_order
        }
        
        dependencies = {
            name: list(self.graph.predecessors(name))
            for name in execution_order
        }
        
        data_flow_map = self._build_data_flow_map()
        
        return PipelineExecutionPlan(
            execution_order=execution_order,
            step_configs=step_configs,
            dependencies=dependencies,
            data_flow_map=data_flow_map
        )
    
    def _build_data_flow_map(self) -> Dict[str, Dict[str, str]]:
        """Map data flow between steps."""
        data_flow = {}
        
        for step_name in self.graph.nodes():
            step_config = self.graph.nodes[step_name]['config']
            inputs = {}
            
            # Map input dependencies
            for dep_step in self.graph.predecessors(step_name):
                dep_config = self.graph.nodes[dep_step]['config']
                # Map outputs of dependency to inputs of current step
                for output_key, output_path in dep_config.outputs.items():
                    if output_key in step_config.inputs:
                        inputs[output_key] = f"{dep_step}:{output_key}"
            
            data_flow[step_name] = inputs
        
        return data_flow

# src/cursus/testing/pipeline_executor.py
class PipelineExecutor:
    """Executes entire pipeline with data flow validation."""
    
    def __init__(self, workspace_dir: str = "./test_workspace"):
        self.workspace_dir = Path(workspace_dir)
        self.script_executor = PipelineScriptExecutor(workspace_dir)
        self.data_validator = DataCompatibilityValidator()
        self.execution_results = {}
    
    def execute_pipeline(self, dag: PipelineDAG, 
                        data_source: str = "synthetic") -> PipelineExecutionResult:
        """Execute complete pipeline with data flow validation."""
        resolver = PipelineDAGResolver(dag)
        execution_plan = resolver.create_execution_plan()
        
        results = []
        step_outputs = {}
        
        for step_name in execution_plan.execution_order:
            try:
                # Prepare step inputs from previous outputs
                step_inputs = self._prepare_step_inputs(
                    step_name, execution_plan, step_outputs
                )
                
                # Execute step
                step_result = self.script_executor.execute_step(
                    step_name, 
                    execution_plan.step_configs[step_name],
                    step_inputs
                )
                
                # Validate outputs
                self._validate_step_outputs(step_result)
                
                # Store outputs for next steps
                step_outputs[step_name] = step_result.outputs
                results.append(step_result)
                
            except Exception as e:
                return PipelineExecutionResult(
                    success=False,
                    error=f"Pipeline failed at step {step_name}: {str(e)}",
                    completed_steps=results,
                    execution_plan=execution_plan
                )
        
        return PipelineExecutionResult(
            success=True,
            completed_steps=results,
            execution_plan=execution_plan,
            total_duration=sum(r.duration for r in results)
        )
```

### Day 3-4: Data Flow Validation
```python
# src/cursus/testing/data_flow_validator.py
from typing import Any, Dict, List, Optional
from pathlib import Path
import pandas as pd
import json

@dataclass
class DataCompatibilityReport:
    """Report on data compatibility between steps."""
    compatible: bool
    issues: List[str]
    warnings: List[str]
    data_summary: Dict[str, Any]

class DataCompatibilityValidator:
    """Validates data compatibility between pipeline steps."""
    
    def __init__(self):
        self.compatibility_rules = self._load_compatibility_rules()
    
    def validate_step_transition(self, 
                               producer_output: Dict[str, Any],
                               consumer_input_spec: Dict[str, Any]) -> DataCompatibilityReport:
        """Validate data compatibility between producer and consumer."""
        issues = []
        warnings = []
        
        # Check required files exist
        for required_file in consumer_input_spec.get('required_files', []):
            if required_file not in producer_output.get('files', {}):
                issues.append(f"Missing required file: {required_file}")
        
        # Check data formats
        for file_name, file_info in producer_output.get('files', {}).items():
            expected_format = consumer_input_spec.get('file_formats', {}).get(file_name)
            if expected_format and file_info.get('format') != expected_format:
                issues.append(
                    f"Format mismatch for {file_name}: "
                    f"expected {expected_format}, got {file_info.get('format')}"
                )
        
        # Check data schemas
        schema_issues = self._validate_schemas(producer_output, consumer_input_spec)
        issues.extend(schema_issues)
        
        return DataCompatibilityReport(
            compatible=len(issues) == 0,
            issues=issues,
            warnings=warnings,
            data_summary=self._create_data_summary(producer_output)
        )
    
    def _validate_schemas(self, output: Dict[str, Any], 
                         input_spec: Dict[str, Any]) -> List[str]:
        """Validate data schemas match expectations."""
        issues = []
        
        for file_name, file_info in output.get('files', {}).items():
            expected_schema = input_spec.get('schemas', {}).get(file_name)
            if expected_schema and 'schema' in file_info:
                actual_schema = file_info['schema']
                
                # Check required columns
                required_cols = expected_schema.get('required_columns', [])
                actual_cols = actual_schema.get('columns', [])
                
                missing_cols = set(required_cols) - set(actual_cols)
                if missing_cols:
                    issues.append(
                        f"Missing columns in {file_name}: {missing_cols}"
                    )
                
                # Check data types
                for col, expected_type in expected_schema.get('column_types', {}).items():
                    actual_type = actual_schema.get('column_types', {}).get(col)
                    if actual_type and actual_type != expected_type:
                        issues.append(
                            f"Type mismatch in {file_name}.{col}: "
                            f"expected {expected_type}, got {actual_type}"
                        )
        
        return issues
```

### Day 5: Error Handling and Recovery
```python
# src/cursus/testing/error_handling.py
from enum import Enum
from typing import Optional, Callable, Any

class ErrorSeverity(Enum):
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class RecoveryStrategy(Enum):
    SKIP_STEP = "skip_step"
    RETRY_WITH_DEFAULTS = "retry_with_defaults"
    USE_CACHED_OUTPUT = "use_cached_output"
    FAIL_PIPELINE = "fail_pipeline"

@dataclass
class ErrorContext:
    """Context information for error handling."""
    step_name: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    suggested_recovery: RecoveryStrategy
    context_data: Dict[str, Any]

class PipelineErrorHandler:
    """Handles errors during pipeline execution."""
    
    def __init__(self):
        self.error_handlers = {
            "import_error": self._handle_import_error,
            "data_compatibility_error": self._handle_data_error,
            "execution_error": self._handle_execution_error,
            "validation_error": self._handle_validation_error
        }
        self.recovery_strategies = {
            RecoveryStrategy.SKIP_STEP: self._skip_step,
            RecoveryStrategy.RETRY_WITH_DEFAULTS: self._retry_with_defaults,
            RecoveryStrategy.USE_CACHED_OUTPUT: self._use_cached_output,
            RecoveryStrategy.FAIL_PIPELINE: self._fail_pipeline
        }
    
    def handle_error(self, error_context: ErrorContext) -> Optional[Any]:
        """Handle error based on context and return recovery result."""
        handler = self.error_handlers.get(error_context.error_type)
        if handler:
            recovery_strategy = handler(error_context)
            return self.recovery_strategies[recovery_strategy](error_context)
        
        # Default to failing pipeline for unknown errors
        return self._fail_pipeline(error_context)
    
    def _handle_import_error(self, context: ErrorContext) -> RecoveryStrategy:
        """Handle script import errors."""
        if "module not found" in context.error_message.lower():
            return RecoveryStrategy.SKIP_STEP
        return RecoveryStrategy.FAIL_PIPELINE
    
    def _handle_data_error(self, context: ErrorContext) -> RecoveryStrategy:
        """Handle data compatibility errors."""
        if context.severity == ErrorSeverity.WARNING:
            return RecoveryStrategy.RETRY_WITH_DEFAULTS
        return RecoveryStrategy.SKIP_STEP
```

## Week 4: Integration and Monitoring

### Day 6-7: Cursus System Integration
```python
# src/cursus/testing/cursus_integration.py
from cursus.core.config_manager import ConfigManager
from cursus.core.contract_manager import ContractManager
from cursus.steps.step_builder_registry import StepBuilderRegistry

class CursusIntegrationLayer:
    """Integration layer with existing Cursus components."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.contract_manager = ContractManager()
        self.step_registry = StepBuilderRegistry()
    
    def resolve_step_configuration(self, step_name: str, 
                                 pipeline_config: Dict[str, Any]) -> StepConfig:
        """Resolve step configuration using Cursus config system."""
        # Get base configuration from registry
        step_builder = self.step_registry.get_builder(step_name)
        base_config = step_builder.get_default_config()
        
        # Apply pipeline-specific overrides
        pipeline_overrides = pipeline_config.get('steps', {}).get(step_name, {})
        merged_config = self.config_manager.merge_configs(
            base_config, pipeline_overrides
        )
        
        return StepConfig.from_dict(merged_config)
    
    def validate_step_contract(self, step_name: str, 
                             config: StepConfig) -> ContractValidationResult:
        """Validate step configuration against contract."""
        contract = self.contract_manager.get_contract(step_name)
        return self.contract_manager.validate_config(config, contract)
    
    def get_script_path(self, step_config: StepConfig) -> Path:
        """Resolve script path from step configuration."""
        if hasattr(step_config, 'script_path') and step_config.script_path:
            return Path(step_config.script_path)
        
        # Fallback to entry_point + source_dir
        if hasattr(step_config, 'entry_point') and hasattr(step_config, 'source_dir'):
            return Path(step_config.source_dir) / step_config.entry_point
        
        raise ValueError(f"Cannot resolve script path for step configuration")
```

### Day 8-9: Pipeline Monitoring and Visualization
```python
# src/cursus/testing/pipeline_monitor.py
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class StepMetrics:
    """Metrics for individual step execution."""
    step_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None
    input_data_size: Optional[int] = None
    output_data_size: Optional[int] = None
    status: str = "running"

@dataclass
class PipelineMetrics:
    """Comprehensive pipeline execution metrics."""
    pipeline_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration: Optional[float] = None
    step_metrics: Dict[str, StepMetrics] = field(default_factory=dict)
    success_rate: float = 0.0
    error_count: int = 0
    warning_count: int = 0

class PipelineMonitor:
    """Monitors pipeline execution and collects metrics."""
    
    def __init__(self):
        self.active_pipelines: Dict[str, PipelineMetrics] = {}
        self.completed_pipelines: List[PipelineMetrics] = []
    
    def start_pipeline_monitoring(self, pipeline_name: str) -> str:
        """Start monitoring a pipeline execution."""
        pipeline_id = f"{pipeline_name}_{int(time.time())}"
        
        self.active_pipelines[pipeline_id] = PipelineMetrics(
            pipeline_name=pipeline_name,
            start_time=datetime.now()
        )
        
        return pipeline_id
    
    def start_step_monitoring(self, pipeline_id: str, step_name: str):
        """Start monitoring a step execution."""
        if pipeline_id in self.active_pipelines:
            step_metrics = StepMetrics(
                step_name=step_name,
                start_time=datetime.now()
            )
            self.active_pipelines[pipeline_id].step_metrics[step_name] = step_metrics
    
    def complete_step_monitoring(self, pipeline_id: str, step_name: str, 
                               success: bool, metrics: Dict[str, Any]):
        """Complete step monitoring with final metrics."""
        if pipeline_id in self.active_pipelines:
            step_metrics = self.active_pipelines[pipeline_id].step_metrics.get(step_name)
            if step_metrics:
                step_metrics.end_time = datetime.now()
                step_metrics.duration = (
                    step_metrics.end_time - step_metrics.start_time
                ).total_seconds()
                step_metrics.status = "success" if success else "failed"
                
                # Update with additional metrics
                step_metrics.memory_usage = metrics.get('memory_usage')
                step_metrics.cpu_usage = metrics.get('cpu_usage')
                step_metrics.input_data_size = metrics.get('input_data_size')
                step_metrics.output_data_size = metrics.get('output_data_size')
    
    def generate_report(self, pipeline_id: str) -> Dict[str, Any]:
        """Generate comprehensive execution report."""
        if pipeline_id not in self.active_pipelines:
            return {"error": "Pipeline not found"}
        
        pipeline_metrics = self.active_pipelines[pipeline_id]
        
        return {
            "pipeline_name": pipeline_metrics.pipeline_name,
            "execution_summary": {
                "total_duration": pipeline_metrics.total_duration,
                "steps_completed": len([
                    s for s in pipeline_metrics.step_metrics.values() 
                    if s.status == "success"
                ]),
                "steps_failed": len([
                    s for s in pipeline_metrics.step_metrics.values() 
                    if s.status == "failed"
                ]),
                "success_rate": pipeline_metrics.success_rate
            },
            "step_details": {
                name: {
                    "duration": metrics.duration,
                    "status": metrics.status,
                    "memory_usage": metrics.memory_usage,
                    "data_processed": {
                        "input_size": metrics.input_data_size,
                        "output_size": metrics.output_data_size
                    }
                }
                for name, metrics in pipeline_metrics.step_metrics.items()
            },
            "performance_insights": self._generate_performance_insights(pipeline_metrics)
        }
```

### Day 10: Testing and Integration
```python
# test/integration/test_pipeline_execution.py
import pytest
from pathlib import Path
from cursus.testing.pipeline_executor import PipelineExecutor
from cursus.testing.pipeline_dag_resolver import PipelineDAGResolver
from cursus.core.dag import PipelineDAG

class TestPipelineExecution:
    """Integration tests for pipeline execution."""
    
    @pytest.fixture
    def sample_pipeline_dag(self):
        """Create sample pipeline DAG for testing."""
        return PipelineDAG.from_yaml("test/fixtures/sample_pipeline.yaml")
    
    @pytest.fixture
    def pipeline_executor(self, tmp_path):
        """Create pipeline executor with temporary workspace."""
        return PipelineExecutor(workspace_dir=str(tmp_path))
    
    def test_simple_pipeline_execution(self, sample_pipeline_dag, pipeline_executor):
        """Test execution of simple linear pipeline."""
        result = pipeline_executor.execute_pipeline(
            sample_pipeline_dag, 
            data_source="synthetic"
        )
        
        assert result.success
        assert len(result.completed_steps) == len(sample_pipeline_dag.steps)
        assert result.total_duration > 0
    
    def test_pipeline_with_data_validation(self, sample_pipeline_dag, pipeline_executor):
        """Test pipeline execution with data validation."""
        result = pipeline_executor.execute_pipeline(
            sample_pipeline_dag,
            data_source="synthetic"
        )
        
        # Verify data flow validation occurred
        for step_result in result.completed_steps:
            assert step_result.data_validation_report is not None
            if not step_result.data_validation_report.compatible:
                assert len(step_result.data_validation_report.issues) > 0
    
    def test_pipeline_error_recovery(self, pipeline_executor):
        """Test pipeline error handling and recovery."""
        # Create pipeline with intentional error
        faulty_dag = self._create_faulty_pipeline()
        
        result = pipeline_executor.execute_pipeline(faulty_dag)
        
        # Should handle error gracefully
        assert not result.success
        assert result.error is not None
        assert len(result.completed_steps) > 0  # Some steps should complete
```

## Success Metrics

### Week 3 Completion Criteria
- [ ] Pipeline DAG resolver correctly handles topological sorting
- [ ] Data flow validation identifies compatibility issues
- [ ] Error handling system recovers from common failures
- [ ] Integration tests pass for basic pipeline execution

### Week 4 Completion Criteria
- [ ] Cursus system integration works with existing components
- [ ] Pipeline monitoring captures comprehensive metrics
- [ ] Visualization components display execution progress
- [ ] End-to-end integration tests demonstrate full functionality

## Deliverables

1. **Pipeline Execution Engine**
   - PipelineDAGResolver with topological sorting
   - PipelineExecutor with data flow validation
   - DataCompatibilityValidator with schema checking

2. **Error Handling System**
   - PipelineErrorHandler with recovery strategies
   - Comprehensive error classification and handling
   - Graceful degradation for pipeline failures

3. **System Integration**
   - CursusIntegrationLayer for existing component integration
   - Configuration and contract validation integration
   - Script path resolution from step configurations

4. **Monitoring and Visualization**
   - PipelineMonitor with comprehensive metrics collection
   - Performance insights and reporting capabilities
   - Real-time execution progress tracking

5. **Testing Suite**
   - Integration tests for pipeline execution
   - Data validation test scenarios
   - Error handling and recovery test cases

## Risk Mitigation

### Technical Risks
- **Complex DAG Resolution**: Implement incremental testing with simple DAGs first
- **Data Compatibility Issues**: Create comprehensive test data scenarios
- **Integration Complexity**: Use adapter pattern for existing Cursus components

### Timeline Risks
- **Scope Creep**: Focus on core functionality, defer advanced features
- **Integration Delays**: Parallel development of components where possible
- **Testing Complexity**: Automated test generation for common scenarios

## Handoff to Next Phase

### Prerequisites for S3 Integration Phase
1. Pipeline execution engine fully functional
2. Data flow validation system operational
3. Error handling and recovery mechanisms tested
4. Integration with Cursus components verified
5. Monitoring and visualization capabilities demonstrated

### Documentation Requirements
1. API documentation for all new components
2. Integration guide for Cursus system components
3. Error handling and recovery strategy documentation
4. Performance monitoring and metrics guide
5. Testing framework documentation and examples

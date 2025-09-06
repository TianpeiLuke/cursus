---
tags:
  - api_reference
  - sagemaker
  - pipeline
  - cursus
  - documentation
  - implementation
keywords:
  - SageMaker pipeline API
  - MODS pipeline API
  - Cursus API reference
  - pipeline compiler
  - DAG compiler
  - step builders
  - configuration management
  - dependency resolution
topics:
  - SageMaker pipeline API
  - Cursus core components
  - pipeline development
  - ML workflow automation
language: python
date of note: 2025-09-06
---

# SageMaker/MODS Pipeline API Reference

## Overview

This reference documents the complete API for building SageMaker/MODS pipelines using the Cursus package. It covers all core components, from DAG creation to pipeline execution, with practical examples and advanced usage patterns.

## Core API Components

### PipelineDAGCompiler

The main entry point for compiling DAGs into SageMaker pipelines.

```python
from cursus.core.compiler.dag_compiler import PipelineDAGCompiler
```

**Constructor:**
```python
def __init__(
    self,
    config_path: str,
    sagemaker_session: PipelineSession,
    role: str,
    pipeline_name: Optional[str] = None,
    pipeline_description: Optional[str] = None
)
```

**Parameters:**
- `config_path`: Path to the pipeline configuration JSON file
- `sagemaker_session`: SageMaker pipeline session for AWS operations
- `role`: IAM role ARN for pipeline execution
- `pipeline_name`: Optional custom pipeline name
- `pipeline_description`: Optional pipeline description

**Example:**
```python
from sagemaker.workflow.pipeline_context import PipelineSession

pipeline_session = PipelineSession()
compiler = PipelineDAGCompiler(
    config_path="configs/pipeline_config.json",
    sagemaker_session=pipeline_session,
    role="arn:aws:iam::123456789012:role/SageMakerExecutionRole"
)
```

### PipelineDAG

Represents the directed acyclic graph structure of your pipeline.

```python
from cursus.api.dag.base_dag import PipelineDAG
```

**Core Methods:**

#### add_node()
```python
def add_node(self, node_id: str) -> None
```
Adds a step node to the DAG.

**Example:**
```python
dag = PipelineDAG()
dag.add_node("data_loading")
dag.add_node("preprocessing")
dag.add_node("training")
```

#### add_edge()
```python
def add_edge(self, from_node: str, to_node: str) -> None
```
Adds a dependency edge between two nodes.

**Example:**
```python
dag.add_edge("data_loading", "preprocessing")
dag.add_edge("preprocessing", "training")
```

#### Properties
```python
@property
def nodes(self) -> List[str]
    """Returns list of all nodes in the DAG"""

@property  
def edges(self) -> List[Tuple[str, str]]
    """Returns list of all edges as (from_node, to_node) tuples"""
```

## Compilation Methods

### validate_dag_compatibility()

Validates that a DAG can be compiled with available configurations.

```python
def validate_dag_compatibility(self, dag: PipelineDAG) -> ValidationResult
```

**Returns:** `ValidationResult` with validation status and details

**Example:**
```python
validation = compiler.validate_dag_compatibility(dag)

if validation.is_valid:
    print(f"✅ Validation passed (confidence: {validation.avg_confidence:.2f})")
else:
    print("❌ Validation failed:")
    for issue in validation.issues:
        print(f"  - {issue}")
```

### preview_resolution()

Previews how DAG nodes will be resolved to configuration types.

```python
def preview_resolution(self, dag: PipelineDAG) -> ResolutionPreview
```

**Returns:** `ResolutionPreview` with node-to-config mappings and confidence scores

**Example:**
```python
preview = compiler.preview_resolution(dag)

print("Step Resolution Preview:")
for node, config_type in preview.node_config_map.items():
    confidence = preview.resolution_confidence.get(node, 0.0)
    print(f"  {node} → {config_type} (confidence: {confidence:.2f})")
```

### compile_with_report()

Compiles a DAG into a SageMaker pipeline with detailed reporting.

```python
def compile_with_report(self, dag: PipelineDAG) -> Tuple[Pipeline, CompilationReport]
```

**Returns:** Tuple of (SageMaker Pipeline, Compilation Report)

**Example:**
```python
pipeline, report = compiler.compile_with_report(dag)

print(f"Pipeline '{pipeline.name}' compiled successfully!")
print(f"Steps created: {len(pipeline.steps)}")
print(f"Average confidence: {report.avg_confidence:.2f}")

# Access detailed step information
for step_name, step_info in report.step_details.items():
    print(f"  {step_name}: {step_info.builder_type}")
```

### create_execution_document()

Creates a complete execution document with all required parameters.

```python
def create_execution_document(self, user_params: Dict[str, Any]) -> Dict[str, Any]
```

**Parameters:**
- `user_params`: User-provided parameter values

**Returns:** Complete execution document ready for pipeline execution

**Example:**
```python
user_params = {
    "training_dataset": "s3://my-bucket/training-data/",
    "model_output": "s3://my-bucket/model-artifacts/"
}

execution_doc = compiler.create_execution_document(user_params)

# Use with pipeline execution
execution = pipeline.start(execution_input=execution_doc)
```

## Data Models

### ValidationResult

Result of DAG validation operations.

```python
class ValidationResult:
    is_valid: bool
    avg_confidence: float
    issues: List[str]
    missing_configs: List[str]
    unresolvable_builders: List[str]
    config_errors: List[str]
    dependency_issues: List[str]
```

**Example Usage:**
```python
validation = compiler.validate_dag_compatibility(dag)

if not validation.is_valid:
    if validation.missing_configs:
        print(f"Missing configs: {validation.missing_configs}")
    if validation.unresolvable_builders:
        print(f"Unresolvable builders: {validation.unresolvable_builders}")
```

### ResolutionPreview

Preview of how DAG nodes will be resolved.

```python
class ResolutionPreview:
    node_config_map: Dict[str, str]
    resolution_confidence: Dict[str, float]
    unresolved_nodes: List[str]
    warnings: List[str]
```

### CompilationReport

Detailed report of pipeline compilation process.

```python
class CompilationReport:
    success: bool
    pipeline_name: str
    step_count: int
    avg_confidence: float
    step_details: Dict[str, StepCompilationInfo]
    warnings: List[str]
    errors: List[str]
```

## Pipeline Catalog Integration

### Pre-built Pipeline Functions

#### create_pipeline() - XGBoost E2E

Creates a complete XGBoost end-to-end pipeline.

```python
from cursus.pipeline_catalog.pipelines.xgb_e2e_comprehensive import create_pipeline

pipeline, report, compiler, template = create_pipeline(
    config_path="configs/xgb_config.json",
    session=pipeline_session,
    role=role,
    pipeline_name="MyXGBoostPipeline",
    pipeline_description="Custom XGBoost pipeline",
    validate=True
)
```

**Returns:**
- `pipeline`: Compiled SageMaker Pipeline
- `report`: Compilation report with details
- `compiler`: PipelineDAGCompiler instance
- `template`: Pipeline template for further operations

#### create_dag() - XGBoost DAG

Creates the DAG structure for XGBoost pipelines.

```python
from cursus.pipeline_catalog.shared_dags.xgboost.complete_e2e_dag import create_xgboost_complete_e2e_dag

dag = create_xgboost_complete_e2e_dag()
print(f"Created DAG with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
```

## Configuration Management

### Configuration File Structure

Pipeline configurations are defined in JSON format:

```json
{
  "pipeline_name": "my-pipeline",
  "pipeline_description": "Pipeline description",
  
  "base_config": {
    "region": "us-east-1",
    "role": "arn:aws:iam::123456789012:role/SageMakerRole",
    "bucket": "my-sagemaker-bucket",
    "prefix": "pipeline-prefix"
  },
  
  "data_loading_config": {
    "instance_type": "ml.m5.large",
    "instance_count": 1,
    "volume_size": 30,
    "max_runtime_in_seconds": 3600,
    "input_data_s3_uri": "s3://bucket/input-data",
    "output_data_s3_uri": "s3://bucket/processed-data"
  },
  
  "training_config": {
    "instance_type": "ml.m5.xlarge",
    "instance_count": 1,
    "volume_size": 30,
    "max_runtime_in_seconds": 7200,
    "hyperparameters": {
      "max_depth": 6,
      "learning_rate": 0.1,
      "n_estimators": 100
    }
  }
}
```

### Configuration Validation

```python
# Validate configuration before compilation
try:
    compiler = PipelineDAGCompiler(
        config_path="invalid_config.json",
        sagemaker_session=pipeline_session,
        role=role
    )
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

## Step Builders

### Base Step Builder

All step builders inherit from `StepBuilderBase`:

```python
from cursus.steps.builders.base import StepBuilderBase
from cursus.steps.configs.base import BasePipelineConfig

class CustomStepBuilder(StepBuilderBase):
    def __init__(self, config: BasePipelineConfig, **kwargs):
        super().__init__(config, **kwargs)
        
    def build_step(self, **kwargs) -> Step:
        """Implement step creation logic"""
        pass
```

### Available Step Builders

#### Data Loading
```python
from cursus.steps.builders.cradle_data_loading import CradleDataLoadingStepBuilder
from cursus.steps.configs.cradle_data_loading import CradleDataLoadingConfig

config = CradleDataLoadingConfig(
    instance_type="ml.m5.large",
    input_data_s3_uri="s3://bucket/input-data"
)
builder = CradleDataLoadingStepBuilder(config=config)
step = builder.build_step()
```

#### Preprocessing
```python
from cursus.steps.builders.tabular_preprocessing import TabularPreprocessingStepBuilder
from cursus.steps.configs.tabular_preprocessing import TabularPreprocessingConfig

config = TabularPreprocessingConfig(
    instance_type="ml.m5.xlarge",
    features=["feature1", "feature2"],
    target_column="target"
)
builder = TabularPreprocessingStepBuilder(config=config)
step = builder.build_step()
```

#### Training
```python
from cursus.steps.builders.xgboost_training import XGBoostTrainingStepBuilder
from cursus.steps.configs.xgboost_training import XGBoostTrainingConfig

config = XGBoostTrainingConfig(
    instance_type="ml.m5.xlarge",
    hyperparameters={
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100
    }
)
builder = XGBoostTrainingStepBuilder(config=config)
step = builder.build_step()
```

## Workspace Integration

### Workspace-Aware Development

```python
from cursus.workspace import WorkspaceAPI
from cursus.workspace.core import WorkspaceStepDefinition

# Set up workspace
api = WorkspaceAPI()
result = api.setup_developer_workspace(
    developer_id="developer_name",
    template="ml_pipeline"
)

# Define workspace-based pipeline steps
workspace_config = {
    "data_loading": WorkspaceStepDefinition(
        developer_id="data_team_alice",
        step_name="data_loading",
        step_type="CradleDataLoading",
        config_data={"dataset": "customer_data"}
    ),
    "training": WorkspaceStepDefinition(
        developer_id="ml_team_bob",
        step_name="training",
        step_type="XGBoostTraining",
        config_data={"max_depth": 6}
    )
}
```

### Workspace Pipeline Assembly

```python
from cursus.workspace.core import WorkspacePipelineAssembler

assembler = WorkspacePipelineAssembler(
    dag=dag,
    workspace_config_map=workspace_config,
    workspace_root="development/projects"
)

# Validate components
validation = assembler.validate_workspace_components()
if validation['overall_valid']:
    pipeline = assembler.generate_pipeline("WorkspacePipeline")
```

## Advanced Usage Patterns

### Custom Pipeline Templates

```python
from cursus.core.template.base import PipelineTemplateBase

class CustomPipelineTemplate(PipelineTemplateBase):
    CONFIG_CLASSES = {
        'Base': BasePipelineConfig,
        'DataLoading': CradleDataLoadingConfig,
        'Training': XGBoostTrainingConfig
    }
    
    def _create_pipeline_dag(self) -> PipelineDAG:
        dag = PipelineDAG()
        dag.add_node("data_loading")
        dag.add_node("training")
        dag.add_edge("data_loading", "training")
        return dag
    
    def _create_config_map(self) -> Dict[str, BasePipelineConfig]:
        return {
            "data_loading": self.configs['DataLoading'],
            "training": self.configs['Training']
        }

# Use the template
template = CustomPipelineTemplate(
    config_path="config.json",
    sagemaker_session=pipeline_session,
    role=role
)
pipeline = template.generate_pipeline()
```

### Batch Pipeline Operations

```python
# Create multiple pipelines with different configurations
configs = [
    "config_experiment_1.json",
    "config_experiment_2.json", 
    "config_experiment_3.json"
]

pipelines = []
for config_path in configs:
    pipeline, report, compiler, template = create_pipeline(
        config_path=config_path,
        session=pipeline_session,
        role=role,
        pipeline_name=f"Experiment-{Path(config_path).stem}"
    )
    pipelines.append(pipeline)

# Deploy all pipelines
for pipeline in pipelines:
    pipeline.upsert()
    print(f"Deployed: {pipeline.name}")
```

### Pipeline Execution Management

```python
# Start multiple executions with different parameters
execution_configs = [
    {"dataset": "dataset_a", "learning_rate": 0.1},
    {"dataset": "dataset_b", "learning_rate": 0.05},
    {"dataset": "dataset_c", "learning_rate": 0.2}
]

executions = []
for i, params in enumerate(execution_configs):
    execution_doc = compiler.create_execution_document(params)
    execution = pipeline.start(
        execution_display_name=f"Experiment-{i+1}",
        execution_input=execution_doc
    )
    executions.append(execution)

# Monitor all executions
for execution in executions:
    status = execution.describe()
    print(f"{execution.arn}: {status['PipelineExecutionStatus']}")
```

## Error Handling

### Common Exceptions

```python
from cursus.core.exceptions import (
    ConfigurationError,
    CompilationError,
    ValidationError,
    DependencyResolutionError
)

try:
    pipeline, report = compiler.compile_with_report(dag)
except ConfigurationError as e:
    print(f"Configuration issue: {e}")
except CompilationError as e:
    print(f"Compilation failed: {e}")
except ValidationError as e:
    print(f"Validation error: {e}")
except DependencyResolutionError as e:
    print(f"Dependency resolution failed: {e}")
```

### Error Recovery Patterns

```python
# Graceful error handling with fallbacks
def safe_pipeline_creation(config_path, dag):
    try:
        # Try primary compilation
        compiler = PipelineDAGCompiler(
            config_path=config_path,
            sagemaker_session=pipeline_session,
            role=role
        )
        
        validation = compiler.validate_dag_compatibility(dag)
        if not validation.is_valid:
            print("Validation failed, attempting fixes...")
            # Implement fix logic here
            
        pipeline, report = compiler.compile_with_report(dag)
        return pipeline, report
        
    except ConfigurationError as e:
        print(f"Config error: {e}")
        # Try with default configuration
        return create_default_pipeline(dag)
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None

pipeline, report = safe_pipeline_creation("config.json", dag)
```

## Performance Optimization

### Caching Strategies

```python
# Enable step caching for faster iterations
from sagemaker.workflow.cache_config import CacheConfig

cache_config = CacheConfig(
    enable_caching=True,
    expire_after="7d"  # Cache expires after 7 days
)

# Apply to step builders
builder = XGBoostTrainingStepBuilder(
    config=config,
    cache_config=cache_config
)
```

### Parallel Execution

```python
# Design DAG for maximum parallelism
dag = PipelineDAG()

# Parallel data loading
dag.add_node("data_loading_train")
dag.add_node("data_loading_val")
dag.add_node("data_loading_test")

# Parallel preprocessing
dag.add_node("preprocessing_train")
dag.add_node("preprocessing_val")
dag.add_node("preprocessing_test")

# Connect parallel branches
dag.add_edge("data_loading_train", "preprocessing_train")
dag.add_edge("data_loading_val", "preprocessing_val")
dag.add_edge("data_loading_test", "preprocessing_test")

# Converge for training
dag.add_node("training")
dag.add_edge("preprocessing_train", "training")
dag.add_edge("preprocessing_val", "training")
```

## Monitoring and Debugging

### Pipeline Execution Monitoring

```python
def monitor_pipeline_execution(execution):
    """Monitor pipeline execution with detailed logging"""
    
    while True:
        status = execution.describe()
        current_status = status['PipelineExecutionStatus']
        
        print(f"Pipeline Status: {current_status}")
        
        if current_status in ['Succeeded', 'Failed', 'Stopped']:
            break
            
        # Check individual steps
        steps = execution.list_steps()
        for step in steps:
            step_status = step['StepStatus']
            print(f"  {step['StepName']}: {step_status}")
            
            if step_status == 'Failed':
                print(f"    Failure: {step.get('FailureReason', 'Unknown')}")
        
        time.sleep(30)  # Check every 30 seconds
    
    return current_status

# Use the monitor
final_status = monitor_pipeline_execution(execution)
print(f"Pipeline completed with status: {final_status}")
```

### Debug Information

```python
# Get detailed compilation information
pipeline, report = compiler.compile_with_report(dag)

print("Compilation Debug Info:")
print(f"  Success: {report.success}")
print(f"  Step Count: {report.step_count}")
print(f"  Average Confidence: {report.avg_confidence:.2f}")

if report.warnings:
    print("  Warnings:")
    for warning in report.warnings:
        print(f"    - {warning}")

if report.errors:
    print("  Errors:")
    for error in report.errors:
        print(f"    - {error}")

# Step-by-step details
for step_name, step_info in report.step_details.items():
    print(f"  Step {step_name}:")
    print(f"    Builder: {step_info.builder_type}")
    print(f"    Config: {step_info.config_type}")
    print(f"    Confidence: {step_info.confidence:.2f}")
```

## Integration Examples

### CI/CD Pipeline Integration

```python
def deploy_pipeline_from_ci():
    """Deploy pipeline in CI/CD environment"""
    
    # Load configuration from environment
    config_path = os.environ.get('PIPELINE_CONFIG_PATH')
    pipeline_name = os.environ.get('PIPELINE_NAME')
    
    # Create and validate pipeline
    dag = create_xgboost_complete_e2e_dag()
    
    compiler = PipelineDAGCompiler(
        config_path=config_path,
        sagemaker_session=pipeline_session,
        role=os.environ.get('SAGEMAKER_ROLE')
    )
    
    # Validate before deployment
    validation = compiler.validate_dag_compatibility(dag)
    if not validation.is_valid:
        raise Exception(f"Pipeline validation failed: {validation.issues}")
    
    # Compile and deploy
    pipeline, report = compiler.compile_with_report(dag)
    pipeline.upsert()
    
    print(f"Pipeline {pipeline.name} deployed successfully")
    return pipeline.arn

# Use in CI/CD
if __name__ == "__main__":
    pipeline_arn = deploy_pipeline_from_ci()
    print(f"Deployed pipeline ARN: {pipeline_arn}")
```

### MLOps Integration

```python
def create_mlops_pipeline_with_monitoring():
    """Create pipeline with MLOps monitoring integration"""
    
    # Enhanced configuration with monitoring
    config = {
        "pipeline_name": "mlops-monitored-pipeline",
        "monitoring_config": {
            "enable_model_monitoring": True,
            "data_quality_monitoring": True,
            "model_bias_monitoring": True,
            "model_explainability": True
        },
        "experiment_config": {
            "experiment_name": "mlops-experiment",
            "trial_name": f"trial-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        }
    }
    
    # Create pipeline with monitoring
    dag = create_xgboost_complete_e2e_dag()
    
    # Add monitoring steps
    dag.add_node("model_monitoring_setup")
    dag.add_node("data_quality_baseline")
    dag.add_edge("Registration", "model_monitoring_setup")
    dag.add_edge("Registration", "data_quality_baseline")
    
    # Compile with monitoring
    compiler = PipelineDAGCompiler(
        config_path="mlops_config.json",
        sagemaker_session=pipeline_session,
        role=role
    )
    
    pipeline, report = compiler.compile_with_report(dag)
    return pipeline, report

# Deploy MLOps pipeline
mlops_pipeline, mlops_report = create_mlops_pipeline_with_monitoring()
mlops_pipeline.upsert()
```

## API Reference Summary

| Component | Purpose | Key Methods |
|-----------|---------|-------------|
| `PipelineDAGCompiler` | Compile DAGs to pipelines | `compile_with_report()`, `validate_dag_compatibility()` |
| `PipelineDAG` | Define pipeline structure | `add_node()`, `add_edge()` |
| `StepBuilderBase` | Create pipeline steps | `build_step()`, `create_step()` |
| `WorkspaceAPI` | Workspace management | `setup_developer_workspace()`, `discover_workspace_components()` |
| `ValidationResult` | Validation results | `is_valid`, `avg_confidence`, `issues` |
| `CompilationReport` | Compilation details | `success`, `step_details`, `warnings` |

## Best Practices

1. **Always validate DAGs** before compilation
2. **Use configuration files** for parameter management
3. **Handle exceptions gracefully** in production code
4. **Monitor pipeline executions** for debugging
5. **Leverage workspace features** for team collaboration
6. **Cache expensive operations** for faster iterations
7. **Design for parallelism** when possible
8. **Use meaningful names** for steps and pipelines

For additional examples and advanced patterns, see the [SageMaker Pipeline Quick Start Guide](sagemaker_pipeline_quick_start.md) and explore the pipeline catalog examples in `src/cursus/pipeline_catalog/`.

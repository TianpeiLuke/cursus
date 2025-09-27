---
tags:
  - code
  - sagemaker_pipeline_api
  - pipeline_compiler
  - dag_compiler
  - api_reference
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

**Constructor (Current Implementation):**
```python
def __init__(
    self,
    config_path: str,
    sagemaker_session: Optional[PipelineSession] = None,
    role: Optional[str] = None,
    config_resolver: Optional[StepConfigResolver] = None,
    step_catalog: Optional[StepCatalog] = None,
    pipeline_parameters: Optional[List[Union[str, ParameterString]]] = None,
    **kwargs: Any,
)
```

**Parameters:**
- `config_path`: Path to the pipeline configuration JSON file
- `sagemaker_session`: Optional SageMaker pipeline session for AWS operations
- `role`: Optional IAM role ARN for pipeline execution
- `config_resolver`: Optional custom config resolver for step name resolution
- `step_catalog`: Optional custom step catalog (uses default StepCatalog if not provided)
- `pipeline_parameters`: Optional list of pipeline parameters (defaults to standard MODS parameters)
- `**kwargs`: Additional arguments passed to the compiler

**Default Pipeline Parameters:**
```python
# Default parameters automatically included:
PIPELINE_EXECUTION_TEMP_DIR = ParameterString(name="PipelineExecutionTempDir")
KMS_ENCRYPTION_KEY_PARAM = ParameterString(name="KmsEncryptionKey")
SECURITY_GROUP_ID = ParameterString(name="SecurityGroupId")
VPC_SUBNET = ParameterString(name="VpcSubnet")
```

**Example:**
```python
from sagemaker.workflow.pipeline_context import PipelineSession

pipeline_session = PipelineSession()
compiler = PipelineDAGCompiler(
    config_path="configs/pipeline_config.json",
    sagemaker_session=pipeline_session,
    role="arn:aws:iam::123456789012:role/SageMakerExecutionRole"
)

# With custom pipeline parameters
custom_params = [
    ParameterString(name="CustomDataPath"),
    ParameterString(name="ModelVersion")
]

compiler_with_params = PipelineDAGCompiler(
    config_path="configs/pipeline_config.json",
    sagemaker_session=pipeline_session,
    role="arn:aws:iam::123456789012:role/SageMakerExecutionRole",
    pipeline_parameters=custom_params
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

## Execution Document Generation (Separated Architecture)

### ExecutionDocumentGenerator

**NEW**: Standalone execution document generator, separated from pipeline compilation for specialized processing.

```python
from cursus.mods.exe_doc.generator import ExecutionDocumentGenerator
```

**Constructor:**
```python
def __init__(
    self,
    config_path: str,
    sagemaker_session: Optional[PipelineSession] = None,
    role: Optional[str] = None,
    config_resolver: Optional[StepConfigResolver] = None
)
```

**Parameters:**
- `config_path`: Path to configuration file
- `sagemaker_session`: SageMaker session for AWS operations
- `role`: IAM role for AWS operations
- `config_resolver`: Custom config resolver for step name resolution

**Key Features:**
- **Independent Processing**: Operates separately from pipeline compilation
- **DAG-Based Analysis**: Direct DAG-based parameter extraction
- **Specialized Helpers**: CradleDataLoadingHelper and RegistrationHelper for different step types
- **Automatic Resolution**: Intelligent step-to-config mapping with fallbacks

### fill_execution_document()

Main method for filling execution documents with pipeline metadata.

```python
def fill_execution_document(
    self, 
    dag: PipelineDAG, 
    execution_document: Dict[str, Any]
) -> Dict[str, Any]
```

**Parameters:**
- `dag`: PipelineDAG defining the pipeline structure
- `execution_document`: Execution document to fill

**Returns:** Updated execution document with all required parameters

**Example:**
```python
# Create execution document generator
exe_doc_generator = ExecutionDocumentGenerator(
    config_path="config.json",
    sagemaker_session=pipeline_session,
    role=role
)

# Get default execution document from MODS helper
from mods_workflow_helper.sagemaker_pipeline_helper import SagemakerPipelineHelper
default_execution_doc = SagemakerPipelineHelper.get_pipeline_default_execution_document(pipeline)

# Fill execution document using DAG structure
execution_doc = exe_doc_generator.fill_execution_document(dag, default_execution_doc)

print(f"Processed {len(exe_doc_generator.configs)} configurations")
print(f"Available helpers: {[h.__class__.__name__ for h in exe_doc_generator.helpers]}")
```

### Execution Document Helpers

#### CradleDataLoadingHelper
Specialized helper for Cradle data loading step configurations.

```python
from cursus.mods.exe_doc.cradle_helper import CradleDataLoadingHelper

helper = CradleDataLoadingHelper()

# Check if helper can handle a step
can_handle = helper.can_handle_step(step_name, config)

# Extract step configuration
step_config = helper.extract_step_config(step_name, config)

# Get execution step name
exec_step_name = helper.get_execution_step_name(step_name, config)
```

#### RegistrationHelper
Specialized helper for model registration step configurations.

```python
from cursus.mods.exe_doc.registration_helper import RegistrationHelper

helper = RegistrationHelper()

# Create execution document config with related configs
exec_config = helper.create_execution_doc_config_with_related_configs(
    registration_cfg, payload_cfg, package_cfg
)
```

### Benefits of Separated Architecture

1. **Independent Operation**: Execution document generation doesn't require pipeline compilation
2. **Specialized Processing**: Different helpers for different step types (Cradle, Registration, etc.)
3. **Direct DAG Analysis**: Works directly with DAG structure for parameter extraction
4. **Flexible Integration**: Can be used with any pipeline generation system
5. **Enhanced Debugging**: Clear separation makes troubleshooting easier

**Example Complete Workflow:**
```python
# Step 1: Create pipeline and DAG
dag = create_xgboost_complete_e2e_dag()
compiler = PipelineDAGCompiler(config_path="config.json", ...)
pipeline, report = compiler.compile_with_report(dag)

# Step 2: Create execution document generator (separate from compilation)
exe_doc_generator = ExecutionDocumentGenerator(
    config_path="config.json",
    sagemaker_session=pipeline_session,
    role=role
)

# Step 3: Generate execution document using DAG
default_doc = SagemakerPipelineHelper.get_pipeline_default_execution_document(pipeline)
execution_doc = exe_doc_generator.fill_execution_document(dag, default_doc)

# Step 4: Execute pipeline
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

The configuration system uses a hierarchical structure organized into several key sections:

#### Root Structure
```
root/
├── configuration/
│   ├── processing/
│   │   ├── processing_shared
│   │   └── processing_specific
│   ├── shared
│   └── specific/
│       ├── CradleDataLoading_calibration
│       ├── CradleDataLoading_training
│       ├── Registration
│       └── XGBoostTraining
├── metadata/
│   ├── config_types/
│   │   ├── Base → "BasePipelineConfig"
│   │   ├── CradleDataLoading_calibration → "CradleDataLoadConfig"
│   │   ├── CradleDataLoading_training → "CradleDataLoadConfig"
│   │   ├── Package → "PackageStepConfig"
│   │   ├── Payload → "PayloadConfig"
│   │   ├── Processing → "ProcessingStepConfigBase"
│   │   ├── Registration → "ModelRegistrationConfig"
│   │   ├── TabularPreprocessing_calibration → "TabularPreprocessingConfig"
│   │   ├── TabularPreprocessing_training → "TabularPreprocessingConfig"
│   │   ├── XGBoostModelEval_calibration → "XGBoostModelEvalConfig"
│   │   └── XGBoostTraining → "XGBoostTrainingConfig"
│   ├── created_at → "2025-07-10T17:02:40.028426"
│   └── field_sources/
│       ├── all
│       ├── processing
│       └── specific
```

#### Configuration Hierarchy

**Base Configuration (`BasePipelineConfig`)**
```json
{
  "region": "us-east-1",
  "role": "arn:aws:iam::123456789012:role/SageMakerRole",
  "bucket": "my-sagemaker-bucket",
  "prefix": "pipeline-prefix"
}
```

**Processing Configurations**
- `processing_shared`: Common settings for all processing steps
- `processing_specific`: Step-specific processing configurations

**Step-Specific Configurations**
- `CradleDataLoading_training`: Data loading for training phase
- `CradleDataLoading_calibration`: Data loading for calibration phase
- `XGBoostTraining`: XGBoost model training configuration
- `Registration`: Model registration settings

**Example Complete Configuration Structure:**
```json
{
  "configuration": {
    "shared": {
      "region": "us-east-1",
      "role": "arn:aws:iam::123456789012:role/SageMakerRole",
      "bucket": "my-sagemaker-bucket"
    },
    "processing": {
      "processing_shared": {
        "instance_type": "ml.m5.large",
        "instance_count": 1,
        "volume_size": 30
      }
    },
    "specific": {
      "CradleDataLoading_training": {
        "input_data_s3_uri": "s3://bucket/training-data",
        "output_data_s3_uri": "s3://bucket/processed-training"
      },
      "CradleDataLoading_calibration": {
        "input_data_s3_uri": "s3://bucket/calibration-data",
        "output_data_s3_uri": "s3://bucket/processed-calibration"
      },
      "XGBoostTraining": {
        "instance_type": "ml.m5.xlarge",
        "hyperparameters": {
          "max_depth": 6,
          "learning_rate": 0.1,
          "n_estimators": 100
        }
      },
      "Registration": {
        "model_package_group_name": "xgboost-model-group",
        "approval_status": "PendingManualApproval"
      }
    }
  },
  "metadata": {
    "config_types": {
      "Base": "BasePipelineConfig",
      "CradleDataLoading_training": "CradleDataLoadConfig",
      "CradleDataLoading_calibration": "CradleDataLoadConfig",
      "XGBoostTraining": "XGBoostTrainingConfig",
      "Registration": "ModelRegistrationConfig"
    },
    "field_sources": {
      "all": ["shared", "processing", "specific"],
      "processing": ["processing_shared", "processing_specific"],
      "specific": ["step-specific configurations"]
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

## Related Documentation and Code References

### Core Documentation
- **[SageMaker Pipeline Quick Start Guide](sagemaker_pipeline_quick_start.md)** - Step-by-step tutorial for getting started
- **[Cursus Developer Guide](../../0_developer_guide/README.md)** - Complete development documentation
- **[Workspace-Aware Developer Guide](../../01_developer_guide_workspace_aware/README.md)** - Team collaboration patterns
- **[Design Principles](../../1_design/design_principles.md)** - Architectural foundations
- **[Specification-Driven Design](../../1_design/specification_driven_design.md)** - Core design philosophy

### Implementation References
- **[Pipeline Catalog Examples](../../../src/cursus/pipeline_catalog/)** - Production-ready pipeline implementations
  - **[XGBoost E2E Pipeline](../../../src/cursus/pipeline_catalog/pipelines/xgb_e2e_comprehensive.py)** - Complete implementation example
  - **[Shared DAGs](../../../src/cursus/pipeline_catalog/shared_dags/)** - Reusable DAG patterns
- **[Demo Notebooks](../../../demo/)** - Interactive examples
  - **[Configuration Demo](../../../demo/demo_config.ipynb)** - Configuration system walkthrough
  - **[Pipeline Demo](../../../demo/demo_pipeline.ipynb)** - End-to-end execution example

### Core API Components
- **[DAG Compiler Source](../../../src/cursus/core/compiler/dag_compiler.py)** - Main compilation engine implementation
- **[Pipeline DAG Source](../../../src/cursus/api/dag/base_dag.py)** - DAG structure implementation
- **[Step Builders](../../../src/cursus/steps/builders/)** - All step builder implementations
- **[Step Configurations](../../../src/cursus/steps/configs/)** - Configuration class definitions
- **[Step Specifications](../../../src/cursus/steps/specs/)** - Input/output specification definitions

### Configuration System
- **[Base Configuration](../../../src/cursus/core/base/config_base.py)** - Base configuration classes
- **[Config Field Manager](../../1_design/config_field_manager_refactoring.md)** - Advanced configuration management
- **[Three-Tier Config Design](../../0_developer_guide/three_tier_config_design.md)** - Configuration architecture
- **[Configuration Utils](../../../src/cursus/steps/configs/utils.py)** - Configuration merging and utilities

### Step Development
- **[Adding New Steps Guide](../../0_developer_guide/adding_new_pipeline_step.md)** - Creating custom steps
- **[Step Builder Guide](../../0_developer_guide/step_builder.md)** - Step builder patterns
- **[Script Contract Guide](../../0_developer_guide/script_contract.md)** - Processing script development
- **[Validation Framework](../../0_developer_guide/validation_framework_guide.md)** - Step validation patterns

### Registry and Discovery
- **[Step Registry](../../../src/cursus/registry/)** - Step registration system
- **[Step Catalog Integration Guide](../../0_developer_guide/step_catalog_integration_guide.md)** - Using the step catalog system
- **[Hybrid Registry](../../01_developer_guide_workspace_aware/ws_hybrid_registry_integration.md)** - Workspace registry patterns

### Advanced Features
- **[Dependency Resolution](../../1_design/dependency_resolution_system.md)** - How dependencies are resolved
- **[Pipeline Compiler Design](../../1_design/pipeline_compiler.md)** - Compilation architecture
- **[MODS Integration](../../1_design/mods_dag_compiler_design.md)** - Model Operations integration
- **[Pipeline Runtime](../../1_design/pipeline_runtime_core_engine_design.md)** - Runtime execution patterns

### Testing and Validation
- **[Testing Guide](../../0_developer_guide/script_testability_implementation.md)** - Testing pipeline components
- **[Validation Checklist](../../0_developer_guide/validation_checklist.md)** - Pre-deployment validation
- **[Best Practices](../../0_developer_guide/best_practices.md)** - Development best practices

### CLI and Tools
- **[CLI Commands](../../../src/cursus/cli/)** - Command-line interface
- **[Workspace CLI](../../01_developer_guide_workspace_aware/ws_workspace_cli_reference.md)** - Workspace management tools
- **[Pipeline Catalog CLI](../../0_developer_guide/pipeline_catalog_integration_guide.md)** - Catalog management

### Reference Materials
- **[SageMaker Property Reference](../../0_developer_guide/sagemaker_property_path_reference_database.md)** - Complete property mappings
- **[Hyperparameter Classes](../../0_developer_guide/hyperparameter_class.md)** - Model hyperparameter handling
- **[Common Pitfalls](../../0_developer_guide/common_pitfalls.md)** - Avoiding common issues

For additional examples and advanced patterns, explore the comprehensive documentation and code examples referenced above.

---
tags:
  - tutorial
  - quick_start
  - sagemaker
  - pipeline
  - cursus
  - getting_started
keywords:
  - SageMaker pipeline
  - MODS pipeline
  - Cursus tutorial
  - ML pipeline
  - end-to-end workflow
  - pipeline onboarding
  - XGBoost pipeline
  - machine learning
topics:
  - SageMaker pipeline creation
  - Cursus package usage
  - ML workflow automation
  - pipeline development
language: python
date of note: 2025-09-06
---

# SageMaker/MODS Pipeline Quick Start Tutorial

## Overview

This 30-minute tutorial will get you up and running with the Cursus package to build complete SageMaker/MODS pipelines from end to end. You'll learn how to create, configure, and execute production-ready ML pipelines using Cursus's specification-driven design.

## Prerequisites

- Python 3.8+ environment
- AWS account with SageMaker access
- IAM role with SageMaker permissions
- Basic familiarity with machine learning workflows
- Cursus package installed (`pip install cursus`)

## What You'll Build

By the end of this tutorial, you'll have created a complete XGBoost pipeline that includes:
- Data loading from S3
- Tabular data preprocessing
- XGBoost model training
- Model calibration
- Model packaging and registration
- Model evaluation

## Step 1: Environment Setup (5 minutes)

First, let's set up your environment and verify everything is working:

```python
# Import core Cursus components
from cursus.core.compiler.dag_compiler import PipelineDAGCompiler
from cursus.api.dag.base_dag import PipelineDAG
from cursus.pipeline_catalog.pipelines.xgb_e2e_comprehensive import create_pipeline

# Import SageMaker components
from sagemaker import Session
from sagemaker.workflow.pipeline_context import PipelineSession
import boto3

# Initialize AWS session
boto_session = boto3.Session()
sagemaker_session = Session(boto_session=boto_session)
role = sagemaker_session.get_caller_identity_arn()
pipeline_session = PipelineSession()

print("✅ Environment setup complete!")
print(f"📍 AWS Region: {sagemaker_session.boto_region_name}")
print(f"👤 IAM Role: {role}")
```

**Expected Output:**
```
✅ Environment setup complete!
📍 AWS Region: us-east-1
👤 IAM Role: arn:aws:iam::123456789012:role/SageMakerExecutionRole
```

## Step 2: Create Your First Pipeline Configuration (5 minutes)

Based on the demo notebooks, Cursus uses a sophisticated configuration system with Pydantic models. Let's create configurations following the exact patterns from `demo_config.ipynb`:

```python
import json
from pathlib import Path
from datetime import date

# Import the configuration classes from Cursus
from src.cursus.core.base.config_base import BasePipelineConfig
from src.cursus.steps.configs.config_processing_step_base import ProcessingStepConfigBase
from src.cursus.steps.configs.config_cradle_data_loading_step import CradleDataLoadConfig
from src.cursus.steps.configs.config_tabular_preprocessing_step import TabularPreprocessingConfig
from src.cursus.steps.configs.config_xgboost_training_step import XGBoostTrainingConfig
from src.cursus.steps.configs.config_model_calibration_step import ModelCalibrationConfig
from src.cursus.steps.configs.config_xgboost_model_eval_step import XGBoostModelEvalConfig
from src.cursus.steps.configs.config_package_step import PackageConfig
from src.cursus.steps.configs.config_registration_step import RegistrationConfig
from src.cursus.steps.configs.config_payload_step import PayloadConfig

# Import hyperparameters
from src.cursus.core.base.hyperparameters_base import ModelHyperparameters
from src.cursus.steps.hyperparams.hyperparameters_xgboost import XGBoostModelHyperparameters

# Create configuration directory
config_dir = Path("pipeline_configs")
config_dir.mkdir(exist_ok=True)

# Get current date for pipeline versioning
current_date = date.today().strftime("%Y-%m-%d")

# Step 1: Define Base Hyperparameters (following demo_config.ipynb pattern)
print("📋 Step 1: Setting up hyperparameters...")

# Define your feature lists (customize these for your data)
full_field_list = [
    'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5',
    'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10',
    'categorical_1', 'categorical_2', 'target'
]

cat_field_list = ['categorical_1', 'categorical_2']

tab_field_list = [
    'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5',
    'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10'
]

label_name = 'target'
id_name = 'id'
multiclass_categories = [0, 1]  # Binary classification

# Create base hyperparameters
base_hyperparameter = ModelHyperparameters(
    full_field_list=full_field_list,
    cat_field_list=cat_field_list,
    tab_field_list=tab_field_list,
    label_name=label_name,
    id_name=id_name,
    multiclass_categories=multiclass_categories
)

# Create XGBoost-specific hyperparameters
model_params = {
    "num_round": 300,
    "max_depth": 6,
    "min_child_weight": 1,
    "objective": "binary:logistic",
    "eval_metric": "auc"
}

xgb_hyperparams = XGBoostModelHyperparameters.from_base_hyperparam(
    base_hyperparameter,
    **model_params
)

print(f"✅ Hyperparameters created: {xgb_hyperparams.num_classes} classes, {len(xgb_hyperparams.tab_field_list)} features")

# Step 2: Create Base Configuration (following demo_config.ipynb pattern)
print("\n📋 Step 2: Setting up base configuration...")

# Determine source directory (customize this path)
current_dir = Path.cwd()
source_dir = current_dir / 'scripts'  # Directory containing your processing scripts

base_config = BasePipelineConfig(
    bucket=sagemaker_session.default_bucket(),
    current_date=current_date,
    region="NA",  # or "EU", "FE" based on your region
    aws_region=sagemaker_session.boto_region_name,
    author="tutorial_user",
    role=role,
    service_name="CursusTutorial",
    pipeline_version="1.0.0",
    framework_version="1.7-1",
    py_version="py3",
    source_dir=str(source_dir)
)

print(f"✅ Base config created for region: {base_config.region}")

# Step 3: Create Processing Step Base Configuration
print("\n📋 Step 3: Setting up processing configuration...")

processing_source_dir = source_dir / 'processing'
processing_dict = {
    'processing_source_dir': str(processing_source_dir),
    'processing_instance_type_large': 'ml.m5.12xlarge',
    'processing_instance_type_small': 'ml.m5.4xlarge',
}

processing_step_config = ProcessingStepConfigBase.from_base_config(
    base_config,
    **processing_dict
)

print(f"✅ Processing config created with source dir: {processing_step_config.processing_source_dir}")

# Step 4: Create Individual Step Configurations
print("\n📋 Step 4: Creating step-specific configurations...")

config_list = []

# Add base configs to list
config_list.append(base_config)
config_list.append(processing_step_config)

# Training Tabular Preprocessing Config
training_tabular_preprocessing_config = TabularPreprocessingConfig.from_base_config(
    processing_step_config,
    job_type="training",
    label_name=base_hyperparameter.label_name,
    processing_entry_point="tabular_preprocessing.py"
)
training_tabular_preprocessing_config.use_large_processing_instance = True
config_list.append(training_tabular_preprocessing_config)

# Calibration Tabular Preprocessing Config
calibration_tabular_preprocessing_config = TabularPreprocessingConfig.from_base_config(
    processing_step_config,
    job_type="calibration",
    label_name=base_hyperparameter.label_name,
    processing_entry_point="tabular_preprocessing.py"
)
config_list.append(calibration_tabular_preprocessing_config)

# XGBoost Training Config
training_instance_type = "ml.m5.12xlarge"
training_volume_size = 800
training_entry_point = "xgboost_training.py"

train_dict = {
    'training_instance_type': training_instance_type,
    'training_entry_point': training_entry_point,
    'training_volume_size': training_volume_size,
    'hyperparameters': xgb_hyperparams
}

xgboost_train_config = XGBoostTrainingConfig.from_base_config(
    base_config,
    **train_dict
)
config_list.append(xgboost_train_config)

# Model Calibration Config
model_calibration_config = ModelCalibrationConfig.from_base_config(
    processing_step_config,
    label_field=base_hyperparameter.label_name,
    processing_entry_point='model_calibration.py',
    score_field='prob_class_1',
    is_binary=base_hyperparameter.is_binary,
    num_classes=base_hyperparameter.num_classes,
    score_field_prefix='prob_class_',
    multiclass_categories=[i for i in range(base_hyperparameter.num_classes)]
)
config_list.append(model_calibration_config)

# XGBoost Model Evaluation Config
model_eval_processing_entry_point = 'xgboost_model_evaluation.py'
model_eval_job_type = 'calibration'

previous_processing_config = processing_step_config.model_dump()
previous_processing_config['processing_entry_point'] = model_eval_processing_entry_point
previous_processing_config['use_large_processing_instance'] = True

xgboost_model_eval_config = XGBoostModelEvalConfig(
    **previous_processing_config,
    job_type=model_eval_job_type,
    hyperparameters=xgb_hyperparams,
    xgboost_framework_version=base_config.framework_version
)
config_list.append(xgboost_model_eval_config)

# Package Config
package_config = PackageConfig.from_base_config(processing_step_config)
config_list.append(package_config)

# Registration Config
model_registration_config = RegistrationConfig.from_base_config(
    base_config,
    framework='xgboost',
    inference_entry_point='xgboost_inference.py',
    model_owner="your-team-id",  # Replace with your actual team ID
    model_domain="CursusTutorial",
    model_objective="Tutorial_Model_Demo"
)
config_list.append(model_registration_config)

# Payload Config
payload_config = PayloadConfig.from_base_config(
    processing_step_config,
    model_owner="your-team-id",
    model_domain="CursusTutorial", 
    model_objective="Tutorial_Model_Demo",
    expected_tps=2,
    max_latency_in_millisecond=800
)
config_list.append(payload_config)

print(f"✅ Created {len(config_list)} configuration objects")

# Step 5: Merge and Save Configurations (following demo_config.ipynb pattern)
print("\n📋 Step 5: Merging and saving configurations...")

from src.cursus.steps.configs.utils import merge_and_save_configs

# Create config filename following the demo pattern
MODEL_CLASS = 'xgboost'
service_name = "CursusTutorial"
region = base_config.region

config_filename = f'config_{region}_{MODEL_CLASS}_{service_name}.json'
config_path = config_dir / config_filename

# Merge and save all configurations into one JSON file
merged_config = merge_and_save_configs(config_list, str(config_path))

print(f"✅ Configuration saved to: {config_path}")
print(f"📊 Pipeline will use bucket: {sagemaker_session.default_bucket()}")
print(f"📅 Pipeline date: {current_date}")
print(f"🔧 Total configurations: {len(config_list)}")

# Also save hyperparameters separately (following demo pattern)
hyperparam_filename = f'hyperparameters_{region}_{MODEL_CLASS}.json'
hyperparam_path = config_dir / hyperparam_filename

with open(hyperparam_path, 'w') as f:
    json.dump(xgb_hyperparams.model_dump(), f, indent=2, sort_keys=True)

print(f"💾 Hyperparameters saved to: {hyperparam_path}")
```

## Step 3: Create Your Pipeline DAG (5 minutes)

Now let's create the pipeline structure using Cursus's DAG system. This matches the exact structure from the demo notebooks:

```python
def create_xgboost_pipeline_dag() -> PipelineDAG:
    """
    Create the DAG structure for the XGBoost train-calibrate-evaluate pipeline.
    This matches the structure from demo/demo_pipeline.ipynb
    """
    dag = PipelineDAG()
    
    # Add all nodes - exactly as in the demo notebook
    dag.add_node("CradleDataLoading_training")       # Data load for training
    dag.add_node("TabularPreprocessing_training")    # Tabular preprocessing for training
    dag.add_node("XGBoostTraining")                  # XGBoost training step
    dag.add_node("ModelCalibration_calibration")     # Model calibration step with calibration variant
    dag.add_node("Package")                          # Package step
    dag.add_node("Registration")                     # MIMS registration step
    dag.add_node("Payload")                          # Payload step
    dag.add_node("CradleDataLoading_calibration")    # Data load for calibration
    dag.add_node("TabularPreprocessing_calibration") # Tabular preprocessing for calibration
    dag.add_node("XGBoostModelEval_calibration")     # Model evaluation step
    
    # Training flow
    dag.add_edge("CradleDataLoading_training", "TabularPreprocessing_training")
    dag.add_edge("TabularPreprocessing_training", "XGBoostTraining")
    
    # Calibration flow
    dag.add_edge("CradleDataLoading_calibration", "TabularPreprocessing_calibration")
    
    # Evaluation flow
    dag.add_edge("XGBoostTraining", "XGBoostModelEval_calibration")
    dag.add_edge("TabularPreprocessing_calibration", "XGBoostModelEval_calibration")
    
    # Model calibration flow - depends on model evaluation
    dag.add_edge("XGBoostModelEval_calibration", "ModelCalibration_calibration")
    
    # Output flow
    dag.add_edge("ModelCalibration_calibration", "Package")
    dag.add_edge("XGBoostTraining", "Package")  # Raw model is also input to packaging
    dag.add_edge("XGBoostTraining", "Payload")  # Payload test uses the raw model
    dag.add_edge("Package", "Registration")
    dag.add_edge("Payload", "Registration")
    
    print(f"✅ DAG created with {len(dag.nodes)} nodes and {len(dag.edges)} edges")
    return dag

# Create the DAG
dag = create_xgboost_pipeline_dag()

print("📋 Pipeline steps:")
for i, node in enumerate(dag.nodes, 1):
    print(f"   {i}. {node}")
```

**Expected Output:**
```
✅ DAG created with 10 nodes and 11 edges
📋 Pipeline steps:
   1. CradleDataLoading_training
   2. TabularPreprocessing_training
   3. XGBoostTraining
   4. ModelCalibration_calibration
   5. Package
   6. Registration
   7. Payload
   8. CradleDataLoading_calibration
   9. TabularPreprocessing_calibration
   10. XGBoostModelEval_calibration
```

## Step 4: Compile Your Pipeline (5 minutes)

Use Cursus's compiler to convert your DAG into a SageMaker pipeline, following the exact pattern from the demo:

```python
# Create the DAG compiler - this is the core of Cursus
dag_compiler = PipelineDAGCompiler(
    config_path=str(config_path),
    sagemaker_session=pipeline_session,
    role=role
)

# Validate the DAG before compilation
print("🔍 Validating DAG compatibility...")
validation = dag_compiler.validate_dag_compatibility(dag)

if validation.is_valid:
    print("✅ DAG validation passed!")
    print(f"📊 Confidence score: {validation.avg_confidence:.2f}")
else:
    print("⚠️ DAG validation issues found:")
    if validation.missing_configs:
        print(f"   Missing configs: {validation.missing_configs}")
    if validation.unresolvable_builders:
        print(f"   Unresolvable builders: {validation.unresolvable_builders}")
    if validation.config_errors:
        print(f"   Config errors: {validation.config_errors}")

# Preview the resolution - see how DAG nodes map to configurations
print("\n🔍 Previewing step resolution...")
preview = dag_compiler.preview_resolution(dag)
for node, config_type in preview.node_config_map.items():
    confidence = preview.resolution_confidence.get(node, 0.0)
    print(f"   {node} → {config_type} (confidence: {confidence:.2f})")

if preview.recommendations:
    print("\n💡 Recommendations:")
    for recommendation in preview.recommendations:
        print(f"   - {recommendation}")

# Compile the DAG into a SageMaker pipeline
print("\n🏗️ Compiling pipeline...")
try:
    pipeline, report = dag_compiler.compile_with_report(dag=dag)
    
    # Log report summary
    print(f"✅ Pipeline '{pipeline.name}' compiled successfully!")
    print(f"📊 Conversion complete: {report.summary()}")
    
    # Show detailed resolution
    for node, details in report.resolution_details.items():
        print(f"   {node} → {details['config_type']} ({details['builder_type']})")
    
    print(f"🔧 Steps created: {len(pipeline.steps)}")
    
except Exception as e:
    print(f"❌ Failed to convert DAG to pipeline: {e}")
    raise
```

## Step 5: Prepare Sample Data (Optional - 3 minutes)

If you don't have your own data, let's create some sample data:

```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Generate sample dataset
print("📊 Creating sample dataset...")
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    n_redundant=2,
    n_clusters_per_class=1,
    random_state=42
)

# Create DataFrame
feature_names = [f"feature_{i+1}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Split into training and calibration sets
train_df = df.sample(frac=0.7, random_state=42)
calibration_df = df.drop(train_df.index)

print(f"✅ Sample data created:")
print(f"   📈 Training set: {len(train_df)} samples")
print(f"   📊 Calibration set: {len(calibration_df)} samples")
print(f"   🎯 Features: {len(feature_names)}")

# Save to S3 (optional - uncomment if you want to upload sample data)
# training_s3_path = f"s3://{sagemaker_session.default_bucket()}/cursus-tutorial/input-data/training/data.csv"
# calibration_s3_path = f"s3://{sagemaker_session.default_bucket()}/cursus-tutorial/input-data/calibration/data.csv"
# 
# train_df.to_csv("training_data.csv", index=False)
# calibration_df.to_csv("calibration_data.csv", index=False)
# 
# # Upload to S3
# sagemaker_session.upload_data("training_data.csv", key_prefix="cursus-tutorial/input-data/training")
# sagemaker_session.upload_data("calibration_data.csv", key_prefix="cursus-tutorial/input-data/calibration")
# 
# print(f"📤 Data uploaded to S3:")
# print(f"   📈 Training: {training_s3_path}")
# print(f"   📊 Calibration: {calibration_s3_path}")
```

## Step 6: Create Execution Parameters (3 minutes)

Prepare the parameters needed to execute your pipeline, following the demo pattern:

```python
# Get the pipeline template for execution document handling
pipeline_template = dag_compiler.get_last_template()

# Create default execution document - this follows the MODS pattern
from mods_workflow_helper.sagemaker_pipeline_helper import SagemakerPipelineHelper

default_execution_doc = SagemakerPipelineHelper.get_pipeline_default_execution_document(pipeline)

print("📋 Default execution document structure:")
print(json.dumps(default_execution_doc, indent=2))

# Fill in the execution document using the pipeline template
# This automatically populates all required parameters
execution_doc = pipeline_template.fill_execution_document(default_execution_doc)

print("\n✅ Execution document filled with parameters:")
print(json.dumps(execution_doc, indent=2))

# Save execution document
execution_doc_path = config_dir / f"execution_doc_{pipeline.name}.json"
with open(execution_doc_path, "w") as f:
    json.dump(execution_doc, f, indent=2)

print(f"💾 Execution document saved to: {execution_doc_path}")

# You can also customize specific parameters if needed
custom_params = {
    "training_data_s3_uri": f"s3://{sagemaker_session.default_bucket()}/your-training-data/",
    "calibration_data_s3_uri": f"s3://{sagemaker_session.default_bucket()}/your-calibration-data/"
}

# Update execution document with custom parameters
for key, value in custom_params.items():
    if key in execution_doc:
        execution_doc[key] = value
        print(f"📝 Updated {key}: {value}")
```

## Step 7: Deploy and Execute Your Pipeline (4 minutes)

Now let's deploy your pipeline to SageMaker following the exact pattern from `demo_pipeline.ipynb`:

```python
# Deploy the pipeline to SageMaker
print("🚀 Deploying pipeline to SageMaker...")
try:
    pipeline.upsert()
    print(f"✅ Pipeline '{pipeline.name}' deployed successfully!")
    print(f"🔗 Pipeline ARN: {pipeline.arn}")
    
    # List pipeline steps
    print(f"\n📋 Pipeline contains {len(pipeline.steps)} steps:")
    for i, step in enumerate(pipeline.steps, 1):
        print(f"   {i}. {step.name} ({step.__class__.__name__})")
    
except Exception as e:
    print(f"❌ Pipeline deployment failed: {e}")
    print("💡 Check your AWS permissions and configuration")

# Prepare for Execution Document (following demo_pipeline.ipynb pattern)
print("\n📋 Preparing execution document...")

from mods_workflow_helper.sagemaker_pipeline_helper import SagemakerPipelineHelper, SecurityConfig

# Get default execution document - this follows the exact demo pattern
default_execution_doc = SagemakerPipelineHelper.get_pipeline_default_execution_document(pipeline)

print("📋 Default execution document structure:")
print(json.dumps(default_execution_doc, indent=2))

# Fill in the execution document using the pipeline template
# This automatically populates all required parameters (from demo_pipeline.ipynb)
execution_doc_fill = pipeline_template.fill_execution_document(default_execution_doc)

print("\n✅ Execution document filled with parameters:")
print(json.dumps(execution_doc_fill, indent=2))

# Use the filled execution document
test_execution_doc = execution_doc_fill.copy()

# Save execution document locally (following demo pattern)
exe_doc_json_filename = f"execute_doc_{base_config.pipeline_name}_{base_config.pipeline_version}.json"
exe_doc_file_path = config_dir / exe_doc_json_filename

with open(exe_doc_file_path, 'w') as f:
    json.dump(test_execution_doc, f, indent=2)

print(f"💾 Execution document saved to: {exe_doc_file_path}")

# Optional: Start pipeline execution using MODS helper (exact demo pattern)
start_execution = input("\n🤔 Would you like to start a pipeline execution? (y/n): ").lower().strip()

if start_execution == 'y':
    try:
        print("▶️ Starting pipeline execution using MODS helper...")
        
        # Create security config (following demo_pipeline.ipynb pattern)
        # These values are defined earlier in the demo_pipeline.ipynb
        from secure_ai_sandbox_python_lib.session import Session as SaisSession
        from mods_workflow_helper.utils.secure_session import create_secure_session_config
        
        # Initialize SAIS session (if using SAIS environment)
        sais_session = SaisSession(".")
        
        security_config = SecurityConfig(
            kms_key=sais_session.get_team_owned_bucket_kms_key(),
            security_group=sais_session.sandbox_vpc_security_group(),
            vpc_subnets=sais_session.sandbox_vpc_subnets()
        )
        
        # Alternative: If not using SAIS, you can manually specify:
        # security_config = SecurityConfig(
        #     kms_key="your-kms-key-id",  # Replace with your KMS key
        #     security_group="your-security-group-id",  # Replace with your security group
        #     vpc_subnets=["your-vpc-subnet-id"]  # Replace with your VPC subnet
        # )
        
        # Start execution with security configuration (exact demo pattern)
        execution_result = SagemakerPipelineHelper.start_pipeline_execution(
            pipeline=pipeline,
            secure_config=security_config,
            sagemaker_session=pipeline_session,
            preparation_space_local_root="/tmp",
            pipeline_execution_document=execution_doc_fill
        )
        
        print(f"✅ Pipeline execution started!")
        print(f"📊 Execution result: {execution_result}")
        print(f"🌐 View in AWS Console: https://console.aws.amazon.com/sagemaker/home?region={sagemaker_session.boto_region_name}#/pipelines")
        
    except Exception as e:
        print(f"❌ Pipeline execution failed: {e}")
        print("💡 This might be due to missing data, permissions, or security configuration")
        print("💡 Common issues:")
        print("   - Update security_config with your actual KMS key, security group, and VPC subnet")
        print("   - Ensure your IAM role has necessary permissions")
        print("   - Verify your data is available at the specified S3 locations")
        print("💡 For tutorial purposes, you can skip execution and just verify the pipeline was created")
else:
    print("⏭️ Skipping pipeline execution")
    print("💡 Your pipeline is deployed and ready for execution when you have:")
    print("   - Proper security configuration (KMS key, security group, VPC subnet)")
    print("   - Training and calibration data uploaded to S3")
    print("   - All required IAM permissions configured")

# Show pipeline information for verification
print(f"\n📊 Pipeline Summary:")
print(f"   Name: {pipeline.name}")
print(f"   Steps: {len(pipeline.steps)}")
print(f"   Status: Deployed and ready for execution")
print(f"   Config: {config_path}")
print(f"   Execution Doc: {exe_doc_file_path}")
```

## Step 8: Monitor Your Pipeline (Optional)

If you started an execution, here's how to monitor it:

```python
# Monitor pipeline execution (if started)
if 'execution' in locals():
    print("\n📊 Monitoring pipeline execution...")
    
    # Get execution status
    status = execution.describe()
    print(f"Status: {status['PipelineExecutionStatus']}")
    print(f"Started: {status['CreationTime']}")
    
    # List execution steps
    steps = execution.list_steps()
    print(f"\n📋 Execution steps ({len(steps)} total):")
    
    for step in steps:
        step_name = step['StepName']
        step_status = step['StepStatus']
        
        # Add status emoji
        status_emoji = {
            'Executing': '⏳',
            'Completed': '✅',
            'Failed': '❌',
            'Stopped': '⏹️',
            'Stopping': '⏸️'
        }.get(step_status, '❓')
        
        print(f"   {status_emoji} {step_name}: {step_status}")
        
        # Show failure reason if failed
        if step_status == 'Failed' and 'FailureReason' in step:
            print(f"      💥 Failure: {step['FailureReason']}")

    print(f"\n💡 Monitor your pipeline at:")
    print(f"   🌐 AWS Console: https://console.aws.amazon.com/sagemaker/home?region={sagemaker_session.boto_region_name}#/pipelines")
```

## Understanding What You Built

### Pipeline Architecture

Your pipeline follows Cursus's specification-driven design:

1. **Data Loading Steps**: Load training and calibration data from S3
2. **Preprocessing Steps**: Clean and prepare data for training
3. **Training Step**: Train XGBoost model with specified hyperparameters
4. **Evaluation Step**: Evaluate model performance on calibration data
5. **Calibration Step**: Calibrate model probabilities for better predictions
6. **Packaging Step**: Package model for deployment
7. **Payload Step**: Create test payload for model validation
8. **Registration Step**: Register model in SageMaker Model Registry

### Key Cursus Features Used

- **Specification-Driven Design**: Steps automatically connect based on their input/output specifications
- **DAG Compiler**: Converts high-level DAG into SageMaker pipeline
- **Configuration Management**: Centralized configuration with validation
- **Dependency Resolution**: Automatic resolution of step dependencies
- **Pipeline Templates**: Reusable pipeline patterns

## Next Steps

Congratulations! You've successfully created and deployed your first SageMaker pipeline using Cursus. Here's what you can do next:

### 1. Explore Advanced Features

```python
# Use workspace-aware development
from cursus.workspace import WorkspaceAPI

api = WorkspaceAPI()
result = api.setup_developer_workspace(
    developer_id="your_name",
    template="ml_pipeline"
)
```

### 2. Create Custom Steps

```python
# Create custom step builders
from cursus.steps.builders.base import StepBuilderBase
from cursus.steps.configs.base import BasePipelineConfig

class CustomStepBuilder(StepBuilderBase):
    def build_step(self):
        # Your custom implementation
        pass
```

### 3. Use MODS Pipelines

```python
# Explore MODS (Model Operations Data Science) pipelines
from cursus.pipeline_catalog.mods_pipelines import create_mods_pipeline

mods_pipeline = create_mods_pipeline(
    config_path="mods_config.json",
    session=pipeline_session,
    role=role
)
```

### 4. Advanced Pipeline Patterns

- **Multi-model pipelines**: Train multiple models in parallel
- **A/B testing pipelines**: Compare model variants
- **Batch inference pipelines**: Process large datasets
- **Real-time inference pipelines**: Deploy models for real-time predictions

## Troubleshooting

### Common Issues

**Issue: "Configuration validation failed"**
```python
# Check your configuration file format
with open(config_path) as f:
    config = json.load(f)
    print("Configuration keys:", list(config.keys()))
```

**Issue: "Step resolution failed"**
```python
# Check step specifications and dependencies
validation = dag_compiler.validate_dag_compatibility(dag)
print("Validation details:", validation.details)
```

**Issue: "Pipeline execution failed"**
```python
# Check execution logs
if 'execution' in locals():
    steps = execution.list_steps()
    for step in steps:
        if step['StepStatus'] == 'Failed':
            print(f"Failed step: {step['StepName']}")
            print(f"Reason: {step.get('FailureReason', 'Unknown')}")
```

### Getting Help

- **API Reference**: See [SageMaker Pipeline API Reference](sagemaker_pipeline_api_reference.md)
- **Developer Guide**: Check `slipbox/0_developer_guide/` for detailed documentation
- **Design Documents**: Review `slipbox/1_design/` for architectural details
- **Examples**: Explore `src/cursus/pipeline_catalog/` for more pipeline examples

## Summary

You've successfully:

1. ✅ Set up the Cursus environment
2. ✅ Created a pipeline configuration
3. ✅ Built a pipeline DAG with 10 steps
4. ✅ Compiled the DAG into a SageMaker pipeline
5. ✅ Deployed the pipeline to AWS
6. ✅ Learned to monitor pipeline execution

Your pipeline implements a complete ML workflow using Cursus's specification-driven approach, automatically handling step dependencies and data flow. This foundation enables you to build more complex pipelines and leverage Cursus's advanced features for production ML workflows.

## Additional Resources

- **[SageMaker Pipeline API Reference](sagemaker_pipeline_api_reference.md)** - Complete API documentation
- **[Workspace Quick Start](../workspace/workspace_quick_start.md)** - Learn workspace-aware development
- **[Developer Guide](../../0_developer_guide/README.md)** - Comprehensive development guidelines
- **[Design Documents](../../1_design/)** - Architectural details and design decisions

Happy pipeline building! 🚀

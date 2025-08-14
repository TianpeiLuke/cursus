---
tags:
  - design
  - alignment_validation
  - training_step
  - sagemaker_integration
keywords:
  - training step validation
  - model training patterns
  - hyperparameter validation
  - training input/output validation
  - framework-specific training
topics:
  - training step alignment validation
  - model training patterns
  - SageMaker training validation
language: python
date of note: 2025-08-13
---

# Training Step Alignment Validation Patterns

## Related Documents

### Core Step Type Classification and Patterns
- **[SageMaker Step Type Classification Design](sagemaker_step_type_classification_design.md)** - Complete step type taxonomy and classification system
- **[Training Step Builder Patterns](training_step_builder_patterns.md)** - Training step builder design patterns and implementation guidelines

### Step Type-Aware Validation System
- **[SageMaker Step Type-Aware Unified Alignment Tester Design](sagemaker_step_type_aware_unified_alignment_tester_design.md)** - Main step type-aware validation system design

### Level-Specific Alignment Design Documents
- **[Level 1: Script Contract Alignment Design](level1_script_contract_alignment_design.md)** - Script-contract validation patterns and implementation
- **[Level 2: Contract Specification Alignment Design](level2_contract_specification_alignment_design.md)** - Contract-specification validation patterns
- **[Level 3: Specification Dependency Alignment Design](level3_specification_dependency_alignment_design.md)** - Specification-dependency validation patterns
- **[Level 4: Builder Configuration Alignment Design](level4_builder_configuration_alignment_design.md)** - Builder-configuration validation patterns

### Related Step Type Validation Patterns
- **[Processing Step Alignment Validation Patterns](processing_step_alignment_validation_patterns.md)** - Processing step validation patterns
- **[CreateModel Step Alignment Validation Patterns](createmodel_step_alignment_validation_patterns.md)** - CreateModel step validation patterns
- **[Transform Step Alignment Validation Patterns](transform_step_alignment_validation_patterns.md)** - Transform step validation patterns
- **[RegisterModel Step Alignment Validation Patterns](registermodel_step_alignment_validation_patterns.md)** - RegisterModel step validation patterns
- **[Utility Step Alignment Validation Patterns](utility_step_alignment_validation_patterns.md)** - Utility step validation patterns

## Overview

Training steps in SageMaker are designed for machine learning model training, hyperparameter tuning, and model artifact generation. This document defines the specific alignment validation patterns for Training steps, which require different validation approaches compared to Processing steps due to their focus on model training rather than data transformation.

## Training Step Characteristics

### **Core Purpose**
- **Model Training**: Train machine learning models using various frameworks
- **Hyperparameter Management**: Load and utilize hyperparameters for training
- **Model Persistence**: Save trained models and artifacts
- **Training Metrics**: Generate and log training metrics and evaluation results

### **SageMaker Integration**
- **Step Type**: `TrainingStep`
- **Estimator Types**: `XGBoost`, `PyTorch`, `TensorFlow`, `SKLearn`, `HuggingFace`
- **Input Types**: `TrainingInput` (training data, validation data)
- **Output Types**: Model artifacts, training metrics, checkpoints

## 4-Level Validation Framework for Training Steps

### **Level 1: Script Contract Alignment**
Training scripts must align with their contracts for model training patterns.

#### **Required Script Patterns**
```python
# Hyperparameter loading patterns
import json
with open('/opt/ml/input/config/hyperparameters.json', 'r') as f:
    hyperparameters = json.load(f)

# Data loading patterns
train_data = pd.read_csv('/opt/ml/input/data/training/train.csv')
val_data = pd.read_csv('/opt/ml/input/data/validation/val.csv')

# Model training patterns
model = XGBClassifier(**hyperparameters)
model.fit(X_train, y_train)

# Model evaluation patterns
predictions = model.predict(X_val)
accuracy = accuracy_score(y_val, predictions)

# Model saving patterns
model.save_model('/opt/ml/model/model.xgb')
joblib.dump(model, '/opt/ml/model/model.pkl')
```

#### **Environment Variable Usage**
```python
# Required environment variables for training
model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
train_dir = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
val_dir = os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation')
```

#### **Validation Checks**
- ✅ Hyperparameter loading from SageMaker config
- ✅ Training data loading from SageMaker channels
- ✅ Model training implementation
- ✅ Model evaluation and metrics generation
- ✅ Model saving to SageMaker model directory
- ✅ Environment variable usage for path configuration

### **Level 2: Contract-Specification Alignment**
Training contracts must align with step specifications for training configuration.

#### **Contract Requirements**
```python
TRAINING_CONTRACT = {
    "inputs": {
        "training_data": {
            "type": "TrainingInput",
            "source": "s3://bucket/training/",
            "destination": "/opt/ml/input/data/training"
        },
        "validation_data": {
            "type": "TrainingInput",
            "source": "s3://bucket/validation/",
            "destination": "/opt/ml/input/data/validation"
        }
    },
    "outputs": {
        "model_artifacts": {
            "type": "model_artifacts",
            "source": "/opt/ml/model",
            "destination": "s3://bucket/models/"
        }
    },
    "hyperparameters": {
        "max_depth": 6,
        "eta": 0.3,
        "objective": "binary:logistic",
        "num_round": 100
    },
    "environment_variables": {
        "FRAMEWORK": "xgboost",
        "MODEL_TYPE": "classification"
    }
}
```

#### **Specification Alignment**
```python
TRAINING_SPEC = {
    "step_name": "xgboost-training",
    "estimator_config": {
        "instance_type": "ml.m5.xlarge",
        "instance_count": 1,
        "volume_size_in_gb": 30,
        "max_run": 3600
    },
    "inputs": ["TrainingInput"],
    "outputs": ["model_artifacts"],
    "hyperparameters": {
        "max_depth": 6,
        "eta": 0.3,
        "objective": "binary:logistic"
    },
    "code_location": "s3://bucket/code/"
}
```

#### **Validation Checks**
- ✅ Input types match between contract and specification
- ✅ Output types match between contract and specification
- ✅ Hyperparameters are properly defined and consistent
- ✅ Estimator configuration is complete
- ✅ Training instance configuration is appropriate
- ✅ Code location is specified

### **Level 3: Specification-Dependency Alignment**
Training specifications must align with their dependencies and model pipeline requirements.

#### **Dependency Patterns**
```python
# Training data dependencies
dependencies = {
    "upstream_steps": ["data-preprocessing", "feature-engineering"],
    "input_artifacts": ["processed_training_data", "processed_validation_data"],
    "required_permissions": ["s3:GetObject", "s3:PutObject"],
    "downstream_consumers": ["model-evaluation", "model-registration"]
}
```

#### **Model Pipeline Validation**
```python
# Training pipeline flow
training_pipeline_flow = {
    "data_source": "upstream_step.properties.ProcessingOutputConfig.Outputs['training_data'].S3Output.S3Uri",
    "model_destination": "s3://bucket/models/",
    "evaluation_consumers": ["model-evaluation-step", "model-registration-step"],
    "hyperparameter_source": "hyperparameter-tuning-step"
}

# Model artifact flow
model_artifact_flow = {
    "source": "/opt/ml/model",
    "destination": "s3://bucket/models/",
    "consumers": ["create-model-step", "batch-transform-step"],
    "format": "framework_specific"
}
```

#### **Validation Checks**
- ✅ Upstream data preprocessing steps are satisfied
- ✅ Training data sources are available and accessible
- ✅ Model artifact destinations are configured
- ✅ Downstream model consumers are compatible
- ✅ Hyperparameter sources are valid
- ✅ Permission requirements are met

### **Level 4: Builder-Configuration Alignment**
Training step builders must align with their estimator configuration requirements.

#### **Builder Pattern Requirements**
```python
class TrainingStepBuilder:
    def __init__(self):
        self.estimator = None
        self.inputs = []
        self.hyperparameters = {}
        self.code_location = None
    
    def _create_estimator(self):
        """Create estimator instance with proper configuration"""
        return XGBoost(
            entry_point=self.entry_point,
            framework_version="1.5-1",
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            role=self.role,
            hyperparameters=self.hyperparameters
        )
    
    def _prepare_hyperparameters_file(self):
        """Prepare hyperparameters configuration file"""
        hyperparams_path = "/tmp/hyperparameters.json"
        with open(hyperparams_path, 'w') as f:
            json.dump(self.hyperparameters, f)
        return hyperparams_path
    
    def _prepare_training_inputs(self):
        """Prepare training inputs"""
        return [
            TrainingInput(
                s3_data=self.training_data_uri,
                content_type="text/csv"
            ),
            TrainingInput(
                s3_data=self.validation_data_uri,
                content_type="text/csv"
            )
        ]
    
    def create_step(self):
        """Create training step"""
        return TrainingStep(
            name=self.step_name,
            estimator=self._create_estimator(),
            inputs=self._prepare_training_inputs()
        )
```

#### **Configuration Validation**
```python
# Required builder methods
required_methods = [
    "_create_estimator",
    "_prepare_hyperparameters_file",
    "_prepare_training_inputs",
    "create_step"
]

# Required configuration parameters
required_config = {
    "instance_type": "ml.m5.xlarge",
    "instance_count": 1,
    "role": "arn:aws:iam::account:role/SageMakerRole",
    "training_data_uri": "s3://bucket/training/",
    "validation_data_uri": "s3://bucket/validation/",
    "entry_point": "training_script.py",
    "hyperparameters": {"max_depth": 6, "eta": 0.3}
}
```

#### **Validation Checks**
- ✅ Builder implements required methods
- ✅ Estimator configuration is complete
- ✅ Training input configurations are valid
- ✅ Hyperparameters are properly structured
- ✅ Entry point script is accessible
- ✅ IAM role has required permissions

## Framework-Specific Patterns

### **XGBoost Training**
```python
# XGBoost-specific imports
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor

# Hyperparameter loading
with open('/opt/ml/input/config/hyperparameters.json', 'r') as f:
    hyperparams = json.load(f)

# Data loading for XGBoost
dtrain = xgb.DMatrix('/opt/ml/input/data/training/train.csv')
dval = xgb.DMatrix('/opt/ml/input/data/validation/val.csv')

# Model training
model = xgb.train(
    params=hyperparams,
    dtrain=dtrain,
    num_boost_round=hyperparams.get('num_round', 100),
    evals=[(dval, 'validation')],
    early_stopping_rounds=10
)

# Model saving
model.save_model('/opt/ml/model/model.xgb')
```

### **PyTorch Training**
```python
# PyTorch-specific imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Model definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Training loop
model = Net()
optimizer = optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
criterion = nn.CrossEntropyLoss()

for epoch in range(hyperparams['epochs']):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Model saving
torch.save(model.state_dict(), '/opt/ml/model/model.pth')
```

### **SKLearn Training**
```python
# SKLearn-specific imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Model training
model = RandomForestClassifier(
    n_estimators=hyperparams.get('n_estimators', 100),
    max_depth=hyperparams.get('max_depth', None),
    random_state=42
)

model.fit(X_train, y_train)

# Model evaluation
predictions = model.predict(X_val)
accuracy = accuracy_score(y_val, predictions)

# Model saving
joblib.dump(model, '/opt/ml/model/model.pkl')
```

## Validation Requirements

### **Required Patterns**
```python
TRAINING_VALIDATION_REQUIREMENTS = {
    "script_patterns": {
        "hyperparameter_loading": {
            "keywords": ["hyperparameters.json", "json.load", "config"],
            "paths": ["/opt/ml/input/config/hyperparameters.json"],
            "severity": "ERROR"
        },
        "data_loading": {
            "keywords": ["pd.read_", "DMatrix", "DataLoader"],
            "paths": ["/opt/ml/input/data/training", "/opt/ml/input/data/validation"],
            "severity": "ERROR"
        },
        "model_training": {
            "keywords": ["fit", "train", "xgb.train", "model.fit"],
            "severity": "ERROR"
        },
        "model_saving": {
            "keywords": ["save_model", "torch.save", "joblib.dump"],
            "paths": ["/opt/ml/model"],
            "severity": "ERROR"
        },
        "evaluation": {
            "keywords": ["predict", "evaluate", "accuracy_score", "metrics"],
            "severity": "WARNING"
        }
    },
    "contract_requirements": {
        "inputs": ["TrainingInput"],
        "outputs": ["model_artifacts"],
        "hyperparameters": ["framework_specific"],
        "environment_variables": ["FRAMEWORK", "MODEL_TYPE"]
    },
    "builder_requirements": {
        "methods": ["_create_estimator", "_prepare_hyperparameters_file", "_prepare_training_inputs"],
        "configuration": ["instance_type", "role", "entry_point", "hyperparameters"]
    }
}
```

### **Framework-Specific Requirements**
```python
FRAMEWORK_REQUIREMENTS = {
    "xgboost": {
        "imports": ["xgboost", "xgb"],
        "training_patterns": ["xgb.train", "XGBClassifier", "DMatrix"],
        "saving_patterns": ["save_model", "model.xgb"],
        "hyperparameters": ["max_depth", "eta", "objective", "num_round"]
    },
    "pytorch": {
        "imports": ["torch", "torch.nn"],
        "training_patterns": ["nn.Module", "optimizer", "loss.backward"],
        "saving_patterns": ["torch.save", "state_dict"],
        "hyperparameters": ["learning_rate", "epochs", "batch_size"]
    },
    "sklearn": {
        "imports": ["sklearn"],
        "training_patterns": ["fit", "RandomForestClassifier", "SVC"],
        "saving_patterns": ["joblib.dump", "pickle.dump"],
        "hyperparameters": ["n_estimators", "max_depth", "C"]
    }
}
```

### **Common Issues and Recommendations**

#### **Missing Hyperparameter Loading**
```python
# Issue: Hardcoded hyperparameters instead of loading from config
# Recommendation: Load hyperparameters from SageMaker config
with open('/opt/ml/input/config/hyperparameters.json', 'r') as f:
    hyperparams = json.load(f)
```

#### **Missing Model Saving**
```python
# Issue: Model not saved to SageMaker model directory
# Recommendation: Save model to /opt/ml/model/
model.save_model('/opt/ml/model/model.xgb')
```

#### **Missing Training Data Loading**
```python
# Issue: No training data loading from SageMaker channels
# Recommendation: Load data from SageMaker training channels
train_data = pd.read_csv('/opt/ml/input/data/training/train.csv')
```

#### **Missing Evaluation Patterns**
```python
# Issue: No model evaluation or metrics generation
# Recommendation: Add model evaluation and metrics logging
predictions = model.predict(X_val)
accuracy = accuracy_score(y_val, predictions)
print(f"Validation accuracy: {accuracy}")
```

## Best Practices

### **Hyperparameter Management**
- Always load hyperparameters from SageMaker config file
- Provide default values for optional hyperparameters
- Validate hyperparameter types and ranges
- Log hyperparameters for experiment tracking

### **Data Handling**
- Use SageMaker environment variables for data paths
- Implement proper data validation and preprocessing
- Handle missing or corrupted data gracefully
- Use appropriate data loading methods for each framework

### **Model Training**
- Implement proper training loops with error handling
- Add early stopping and checkpointing for long training jobs
- Log training metrics and progress
- Use validation data for model evaluation

### **Model Persistence**
- Save models in framework-appropriate formats
- Include model metadata and versioning information
- Save additional artifacts (preprocessors, encoders)
- Validate saved models can be loaded correctly

### **Resource Management**
- Configure appropriate instance types for training workload
- Monitor memory and CPU usage during training
- Use distributed training for large datasets
- Implement proper cleanup of temporary files

## Integration with Step Type Enhancement System

### **Training Step Enhancer**
```python
class TrainingStepEnhancer(BaseStepEnhancer):
    def __init__(self):
        super().__init__("Training")
        self.reference_examples = [
            "xgboost_training.py",
            "pytorch_training.py",
            "builder_xgboost_training_step.py"
        ]
        self.framework_validators = {
            "xgboost": self._validate_xgboost_training,
            "pytorch": self._validate_pytorch_training,
            "sklearn": self._validate_sklearn_training
        }
    
    def enhance_validation(self, existing_results, script_name):
        additional_issues = []
        
        # Level 1: Training script patterns
        additional_issues.extend(self._validate_training_script_patterns(script_name))
        
        # Level 2: Training specifications
        additional_issues.extend(self._validate_training_specifications(script_name))
        
        # Level 3: Training dependencies
        additional_issues.extend(self._validate_training_dependencies(script_name))
        
        # Level 4: Training builder patterns
        additional_issues.extend(self._validate_training_builder(script_name))
        
        return self._merge_results(existing_results, additional_issues)
```

### **Framework Detection**
```python
def detect_training_framework(script_content: str) -> Optional[str]:
    """Detect training framework from script content"""
    if 'xgboost' in script_content or 'xgb' in script_content:
        return 'xgboost'
    elif 'torch' in script_content or 'pytorch' in script_content:
        return 'pytorch'
    elif 'sklearn' in script_content:
        return 'sklearn'
    elif 'tensorflow' in script_content or 'tf.' in script_content:
        return 'tensorflow'
    return None
```

## Reference Examples

### **XGBoost Training Script**
```python
# cursus/steps/scripts/xgboost_training.py
import json
import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score

def main():
    # Load hyperparameters
    with open('/opt/ml/input/config/hyperparameters.json', 'r') as f:
        hyperparams = json.load(f)
    
    # Load training data
    train_data = pd.read_csv('/opt/ml/input/data/training/train.csv')
    val_data = pd.read_csv('/opt/ml/input/data/validation/val.csv')
    
    # Prepare data
    X_train = train_data.drop('target', axis=1)
    y_train = train_data['target']
    X_val = val_data.drop('target', axis=1)
    y_val = val_data['target']
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Train model
    model = xgb.train(
        params=hyperparams,
        dtrain=dtrain,
        num_boost_round=hyperparams.get('num_round', 100),
        evals=[(dval, 'validation')],
        early_stopping_rounds=10
    )
    
    # Evaluate model
    predictions = model.predict(dval)
    accuracy = accuracy_score(y_val, predictions > 0.5)
    print(f"Validation accuracy: {accuracy}")
    
    # Save model
    model_path = '/opt/ml/model/model.xgb'
    model.save_model(model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
```

### **Training Step Builder**
```python
# cursus/steps/builders/builder_xgboost_training_step.py
from sagemaker.xgboost import XGBoost
from sagemaker.inputs import TrainingInput
from sagemaker.workflow.steps import TrainingStep
import json

class XGBoostTrainingStepBuilder:
    def __init__(self):
        self.step_name = "xgboost-training"
        self.instance_type = "ml.m5.xlarge"
        self.instance_count = 1
        self.role = None
        self.training_data_uri = None
        self.validation_data_uri = None
        self.entry_point = "xgboost_training.py"
        self.hyperparameters = {
            "max_depth": 6,
            "eta": 0.3,
            "objective": "binary:logistic",
            "num_round": 100
        }
    
    def _create_estimator(self):
        return XGBoost(
            entry_point=self.entry_point,
            framework_version="1.5-1",
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            role=self.role,
            hyperparameters=self.hyperparameters
        )
    
    def _prepare_hyperparameters_file(self):
        hyperparams_path = "/tmp/hyperparameters.json"
        with open(hyperparams_path, 'w') as f:
            json.dump(self.hyperparameters, f)
        return hyperparams_path
    
    def _prepare_training_inputs(self):
        return {
            "training": TrainingInput(
                s3_data=self.training_data_uri,
                content_type="text/csv"
            ),
            "validation": TrainingInput(
                s3_data=self.validation_data_uri,
                content_type="text/csv"
            )
        }
    
    def create_step(self):
        return TrainingStep(
            name=self.step_name,
            estimator=self._create_estimator(),
            inputs=self._prepare_training_inputs()
        )
```

## Conclusion

Training step alignment validation patterns provide comprehensive validation for machine learning model training workflows in SageMaker. The 4-level validation framework ensures proper alignment between training scripts, contracts, specifications, and builders, while framework-specific patterns enable targeted validation for different ML frameworks.

The training validation patterns differ significantly from processing patterns by focusing on:
- Hyperparameter management and loading
- Model training loops and optimization
- Model persistence and artifact generation
- Training metrics and evaluation
- Framework-specific training patterns

This validation pattern serves as a critical component of the step type-aware validation system, ensuring that training workflows follow SageMaker best practices and maintain consistency across different ML frameworks.

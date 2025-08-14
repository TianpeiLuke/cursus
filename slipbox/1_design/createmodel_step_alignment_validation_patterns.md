---
tags:
  - design
  - alignment_validation
  - createmodel_step
  - sagemaker_integration
keywords:
  - createmodel step validation
  - model deployment patterns
  - inference code validation
  - container configuration validation
  - model artifact validation
topics:
  - createmodel step alignment validation
  - model deployment patterns
  - SageMaker model creation validation
language: python
date of note: 2025-08-13
---

# CreateModel Step Alignment Validation Patterns

## Related Documents

### Core Step Type Classification and Patterns
- **[SageMaker Step Type Classification Design](sagemaker_step_type_classification_design.md)** - Complete step type taxonomy and classification system
- **[CreateModel Step Builder Patterns](createmodel_step_builder_patterns.md)** - CreateModel step builder design patterns and implementation guidelines

### Step Type-Aware Validation System
- **[SageMaker Step Type-Aware Unified Alignment Tester Design](sagemaker_step_type_aware_unified_alignment_tester_design.md)** - Main step type-aware validation system design

### Level-Specific Alignment Design Documents
- **[Level 1: Script Contract Alignment Design](level1_script_contract_alignment_design.md)** - Script-contract validation patterns and implementation
- **[Level 2: Contract Specification Alignment Design](level2_contract_specification_alignment_design.md)** - Contract-specification validation patterns
- **[Level 3: Specification Dependency Alignment Design](level3_specification_dependency_alignment_design.md)** - Specification-dependency validation patterns
- **[Level 4: Builder Configuration Alignment Design](level4_builder_configuration_alignment_design.md)** - Builder-configuration validation patterns

### Related Step Type Validation Patterns
- **[Processing Step Alignment Validation Patterns](processing_step_alignment_validation_patterns.md)** - Processing step validation patterns
- **[Training Step Alignment Validation Patterns](training_step_alignment_validation_patterns.md)** - Training step validation patterns
- **[Transform Step Alignment Validation Patterns](transform_step_alignment_validation_patterns.md)** - Transform step validation patterns
- **[RegisterModel Step Alignment Validation Patterns](registermodel_step_alignment_validation_patterns.md)** - RegisterModel step validation patterns
- **[Utility Step Alignment Validation Patterns](utility_step_alignment_validation_patterns.md)** - Utility step validation patterns

## Overview

CreateModel steps in SageMaker are designed for model deployment preparation, inference code packaging, and model endpoint creation. This document defines the specific alignment validation patterns for CreateModel steps, which follow a fundamentally different pattern from Processing and Training steps due to their focus on model deployment rather than script execution.

## CreateModel Step Characteristics

### **Core Purpose**
- **Model Deployment**: Prepare trained models for inference endpoints
- **Inference Code**: Package inference logic and model loading code
- **Container Configuration**: Configure deployment containers and environments
- **Model Artifact Management**: Handle model artifacts and dependencies

### **SageMaker Integration**
- **Step Type**: `CreateModelStep`
- **Model Types**: Framework models (XGBoost, PyTorch, TensorFlow, SKLearn)
- **Input Types**: Model artifacts from training steps
- **Output Types**: SageMaker Model objects for deployment

### **Key Difference from Other Step Types**
- **No Standalone Scripts**: CreateModel steps typically don't have separate execution scripts
- **Builder-Focused**: Validation focuses on step builders rather than script contracts
- **Deployment-Oriented**: Emphasis on container images and inference configuration
- **Artifact-Based**: Works with pre-trained model artifacts rather than training data

## 4-Level Validation Framework for CreateModel Steps

### **Level 1: Builder Configuration Validation** (Replaces Script Validation)
CreateModel builders must implement proper model creation and configuration patterns.

#### **Required Builder Patterns**
```python
class CreateModelStepBuilder:
    def __init__(self):
        self.model_data = None
        self.image_uri = None
        self.role = None
        self.model_name = None
    
    def _create_model(self):
        """Create SageMaker Model instance"""
        return Model(
            model_data=self.model_data,
            image_uri=self.image_uri,
            role=self.role,
            name=self._generate_model_name(),
            env=self._get_environment_variables()
        )
    
    def _generate_model_name(self):
        """Generate unique model name"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{self.base_name}-{timestamp}"
    
    def _get_environment_variables(self):
        """Get model environment variables"""
        return {
            'SAGEMAKER_PROGRAM': 'inference.py',
            'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code'
        }
    
    def create_step(self):
        """Create CreateModel step"""
        return CreateModelStep(
            name=self.step_name,
            model=self._create_model(),
            depends_on=self.dependencies
        )
```

#### **Validation Checks**
- ✅ Builder implements `_create_model` method
- ✅ Model data source is configured
- ✅ Container image URI is specified
- ✅ Execution role is configured
- ✅ Model name generation is implemented
- ✅ Environment variables are set appropriately

### **Level 2: Container and Deployment Configuration Validation**
CreateModel steps must configure proper container images and deployment settings.

#### **Container Configuration Requirements**
```python
CONTAINER_CONFIG = {
    "image_uri": {
        "xgboost": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.5-1",
        "pytorch": "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.12.0-gpu-py38",
        "sklearn": "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.23-1"
    },
    "environment_variables": {
        "SAGEMAKER_PROGRAM": "inference.py",
        "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/code",
        "SAGEMAKER_MODEL_SERVER_TIMEOUT": "3600"
    },
    "instance_types": {
        "cpu": ["ml.t2.medium", "ml.m5.large", "ml.m5.xlarge"],
        "gpu": ["ml.p3.2xlarge", "ml.g4dn.xlarge"]
    }
}
```

#### **Deployment Configuration**
```python
DEPLOYMENT_CONFIG = {
    "model_data_source": {
        "type": "s3_uri",
        "format": "tar.gz",
        "required_files": ["model artifacts", "inference code", "requirements.txt"]
    },
    "inference_configuration": {
        "handler_functions": ["model_fn", "input_fn", "predict_fn", "output_fn"],
        "entry_point": "inference.py",
        "dependencies": "requirements.txt"
    },
    "resource_configuration": {
        "instance_type": "ml.m5.large",
        "initial_instance_count": 1,
        "max_concurrent_transforms": 1
    }
}
```

#### **Validation Checks**
- ✅ Container image URI is valid and accessible
- ✅ Instance type is appropriate for model requirements
- ✅ Environment variables are properly configured
- ✅ Model data source is accessible
- ✅ Inference configuration is complete
- ✅ Resource limits are set appropriately

### **Level 3: Model Artifact Structure Validation**
CreateModel steps must handle model artifacts and inference code properly.

#### **Model Artifact Requirements**
```python
MODEL_ARTIFACT_STRUCTURE = {
    "required_files": {
        "model_file": {
            "xgboost": ["model.xgb", "model.json", "model.pkl"],
            "pytorch": ["model.pth", "model.pt", "pytorch_model.bin"],
            "sklearn": ["model.pkl", "model.joblib"],
            "tensorflow": ["saved_model.pb", "model.h5"]
        },
        "inference_code": ["inference.py", "model_handler.py"],
        "dependencies": ["requirements.txt", "environment.yml"]
    },
    "optional_files": {
        "preprocessing": ["preprocessor.pkl", "scaler.pkl"],
        "metadata": ["model_metadata.json", "feature_names.json"],
        "configuration": ["config.json", "hyperparameters.json"]
    }
}
```

#### **Inference Code Patterns**
```python
# Required inference functions
def model_fn(model_dir):
    """Load and return the model for inference"""
    model_path = os.path.join(model_dir, 'model.xgb')
    model = xgb.Booster()
    model.load_model(model_path)
    return model

def input_fn(request_body, content_type):
    """Parse and preprocess input data"""
    if content_type == 'text/csv':
        return pd.read_csv(StringIO(request_body))
    elif content_type == 'application/json':
        return pd.read_json(StringIO(request_body))
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """Run inference on the model"""
    dmatrix = xgb.DMatrix(input_data)
    predictions = model.predict(dmatrix)
    return predictions

def output_fn(prediction, accept):
    """Format and return the prediction output"""
    if accept == 'application/json':
        return json.dumps(prediction.tolist())
    elif accept == 'text/csv':
        return ','.join(map(str, prediction))
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
```

#### **Validation Checks**
- ✅ Model artifacts are present and in correct format
- ✅ Inference code implements required functions
- ✅ Dependencies are properly specified
- ✅ Model loading logic is framework-appropriate
- ✅ Input/output handling is implemented
- ✅ Error handling is included

### **Level 4: Model Creation Builder Patterns Validation**
CreateModel builders must implement proper step creation and dependency management.

#### **Builder Pattern Requirements**
```python
class CreateModelStepBuilder:
    def __init__(self):
        self.step_name = None
        self.model_data = None
        self.image_uri = None
        self.role = None
        self.dependencies = []
        self.environment_variables = {}
    
    def _create_model(self):
        """Create SageMaker Model with proper configuration"""
        return Model(
            model_data=self.model_data,
            image_uri=self.image_uri,
            role=self.role,
            name=self._generate_model_name(),
            env=self.environment_variables
        )
    
    def _generate_model_name(self):
        """Generate unique model name with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{self.step_name}-model-{timestamp}"
    
    def _configure_dependencies(self):
        """Configure step dependencies"""
        if self.training_step:
            self.dependencies.append(self.training_step)
        return self.dependencies
    
    def create_step(self):
        """Create CreateModel step with dependencies"""
        return CreateModelStep(
            name=self.step_name,
            model=self._create_model(),
            depends_on=self._configure_dependencies()
        )
```

#### **Step Integration Patterns**
```python
# Integration with training step
def integrate_with_training_step(self, training_step):
    """Integrate with upstream training step"""
    self.model_data = training_step.properties.ModelArtifacts.S3ModelArtifacts
    self.dependencies.append(training_step)

# Integration with model registry
def prepare_for_registration(self):
    """Prepare model for registration"""
    self.model_metadata = {
        "model_name": self.model_name,
        "framework": self.framework,
        "created_at": datetime.now().isoformat()
    }

# Integration with batch transform
def prepare_for_batch_transform(self):
    """Prepare model for batch transform"""
    self.batch_transform_config = {
        "instance_type": "ml.m5.large",
        "instance_count": 1,
        "max_concurrent_transforms": 1
    }
```

#### **Validation Checks**
- ✅ Step creation patterns are implemented
- ✅ Model name generation is unique and consistent
- ✅ Dependencies are properly configured
- ✅ Integration with upstream steps is correct
- ✅ Model metadata is prepared
- ✅ Downstream step preparation is included

## Framework-Specific Patterns

### **XGBoost Model Creation**
```python
class XGBoostModelStepBuilder(CreateModelStepBuilder):
    def __init__(self):
        super().__init__()
        self.framework = "xgboost"
        self.image_uri = "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.5-1"
    
    def _create_model(self):
        return Model(
            model_data=self.model_data,
            image_uri=self.image_uri,
            role=self.role,
            name=self._generate_model_name(),
            env={
                'SAGEMAKER_PROGRAM': 'inference.py',
                'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code'
            }
        )

# XGBoost inference code
def model_fn(model_dir):
    """Load XGBoost model"""
    model_path = os.path.join(model_dir, 'model.xgb')
    model = xgb.Booster()
    model.load_model(model_path)
    return model

def predict_fn(input_data, model):
    """XGBoost prediction"""
    dmatrix = xgb.DMatrix(input_data)
    return model.predict(dmatrix)
```

### **PyTorch Model Creation**
```python
class PyTorchModelStepBuilder(CreateModelStepBuilder):
    def __init__(self):
        super().__init__()
        self.framework = "pytorch"
        self.image_uri = "763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-inference:1.12.0-gpu-py38"
    
    def _create_model(self):
        return Model(
            model_data=self.model_data,
            image_uri=self.image_uri,
            role=self.role,
            name=self._generate_model_name(),
            env={
                'SAGEMAKER_PROGRAM': 'inference.py',
                'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code'
            }
        )

# PyTorch inference code
def model_fn(model_dir):
    """Load PyTorch model"""
    model_path = os.path.join(model_dir, 'model.pth')
    model = torch.load(model_path, map_location='cpu')
    model.eval()
    return model

def predict_fn(input_data, model):
    """PyTorch prediction"""
    with torch.no_grad():
        return model(input_data)
```

### **SKLearn Model Creation**
```python
class SKLearnModelStepBuilder(CreateModelStepBuilder):
    def __init__(self):
        super().__init__()
        self.framework = "sklearn"
        self.image_uri = "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-scikit-learn:0.23-1"

# SKLearn inference code
def model_fn(model_dir):
    """Load SKLearn model"""
    model_path = os.path.join(model_dir, 'model.pkl')
    return joblib.load(model_path)

def predict_fn(input_data, model):
    """SKLearn prediction"""
    return model.predict(input_data)
```

## Validation Requirements

### **Required Patterns**
```python
CREATEMODEL_VALIDATION_REQUIREMENTS = {
    "builder_patterns": {
        "model_creation": {
            "keywords": ["_create_model", "Model", "create_model"],
            "severity": "ERROR"
        },
        "model_data_configuration": {
            "keywords": ["model_data", "ModelDataUrl", "model_artifacts"],
            "severity": "ERROR"
        },
        "container_configuration": {
            "keywords": ["image_uri", "container", "image"],
            "severity": "ERROR"
        },
        "step_creation": {
            "keywords": ["create_step", "CreateModelStep"],
            "severity": "ERROR"
        }
    },
    "inference_patterns": {
        "model_loading": {
            "keywords": ["model_fn", "load", "torch.load", "xgb.Booster"],
            "severity": "ERROR"
        },
        "prediction": {
            "keywords": ["predict_fn", "predict", "inference"],
            "severity": "ERROR"
        },
        "input_handling": {
            "keywords": ["input_fn", "preprocess", "parse"],
            "severity": "WARNING"
        },
        "output_handling": {
            "keywords": ["output_fn", "postprocess", "format"],
            "severity": "WARNING"
        }
    },
    "artifact_requirements": {
        "model_files": ["framework_specific"],
        "inference_code": ["inference.py"],
        "dependencies": ["requirements.txt"]
    }
}
```

### **Common Issues and Recommendations**

#### **Missing Model Creation Method**
```python
# Issue: Builder doesn't implement _create_model method
# Recommendation: Add model creation method
def _create_model(self):
    return Model(
        model_data=self.model_data,
        image_uri=self.image_uri,
        role=self.role,
        name=self._generate_model_name()
    )
```

#### **Missing Container Configuration**
```python
# Issue: No container image specified
# Recommendation: Add appropriate container image
self.image_uri = "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.5-1"
```

#### **Missing Inference Code**
```python
# Issue: No inference functions implemented
# Recommendation: Add required inference functions
def model_fn(model_dir):
    # Load model from model_dir
    return model

def predict_fn(input_data, model):
    # Run inference
    return predictions
```

#### **Missing Model Name Generation**
```python
# Issue: Static model names causing conflicts
# Recommendation: Generate unique model names
def _generate_model_name(self):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{self.step_name}-model-{timestamp}"
```

## Best Practices

### **Model Artifact Management**
- Use proper model serialization formats for each framework
- Include all necessary dependencies and requirements
- Validate model artifacts can be loaded correctly
- Include model metadata and versioning information

### **Container Configuration**
- Use official SageMaker container images when possible
- Configure appropriate environment variables
- Set proper resource limits and timeouts
- Use framework-specific container optimizations

### **Inference Code Design**
- Implement all required inference functions
- Add proper error handling and logging
- Optimize for inference performance
- Handle different input/output formats

### **Deployment Preparation**
- Generate unique model names to avoid conflicts
- Configure proper IAM roles and permissions
- Set appropriate instance types for workload
- Prepare for integration with downstream steps

## Integration with Step Type Enhancement System

### **CreateModel Step Enhancer**
```python
class CreateModelStepEnhancer(BaseStepEnhancer):
    def __init__(self):
        super().__init__("CreateModel")
        self.reference_examples = [
            "builder_xgboost_model_step.py",
            "builder_pytorch_model_step.py"
        ]
        self.framework_validators = {
            "xgboost": self._validate_xgboost_model_creation,
            "pytorch": self._validate_pytorch_model_creation
        }
    
    def enhance_validation(self, existing_results, script_name):
        additional_issues = []
        
        # Level 1: Builder configuration validation
        additional_issues.extend(self._validate_builder_configuration(script_name))
        
        # Level 2: Container deployment configuration
        additional_issues.extend(self._validate_container_deployment_configuration(script_name))
        
        # Level 3: Model artifact structure validation
        additional_issues.extend(self._validate_model_artifact_structure(script_name))
        
        # Level 4: Model creation builder patterns
        additional_issues.extend(self._validate_model_creation_builder_patterns(script_name))
        
        return self._merge_results(existing_results, additional_issues)
```

### **Framework Detection**
```python
def detect_createmodel_framework(builder_content: str) -> Optional[str]:
    """Detect framework from CreateModel builder content"""
    if 'xgboost' in builder_content.lower() or 'xgb' in builder_content.lower():
        return 'xgboost'
    elif 'pytorch' in builder_content.lower() or 'torch' in builder_content.lower():
        return 'pytorch'
    elif 'sklearn' in builder_content.lower() or 'scikit' in builder_content.lower():
        return 'sklearn'
    elif 'tensorflow' in builder_content.lower():
        return 'tensorflow'
    return None
```

## Reference Examples

### **XGBoost Model Step Builder**
```python
# cursus/steps/builders/builder_xgboost_model_step.py
from sagemaker.model import Model
from sagemaker.workflow.steps import CreateModelStep
from datetime import datetime

class XGBoostModelStepBuilder:
    def __init__(self):
        self.step_name = "create-xgboost-model"
        self.model_data = None
        self.image_uri = "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.5-1"
        self.role = None
        self.dependencies = []
    
    def _create_model(self):
        return Model(
            model_data=self.model_data,
            image_uri=self.image_uri,
            role=self.role,
            name=self._generate_model_name(),
            env={
                'SAGEMAKER_PROGRAM': 'inference.py',
                'SAGEMAKER_SUBMIT_DIRECTORY': '/opt/ml/code'
            }
        )
    
    def _generate_model_name(self):
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"xgboost-model-{timestamp}"
    
    def integrate_with_training_step(self, training_step):
        self.model_data = training_step.properties.ModelArtifacts.S3ModelArtifacts
        self.dependencies.append(training_step)
    
    def create_step(self):
        return CreateModelStep(
            name=self.step_name,
            model=self._create_model(),
            depends_on=self.dependencies
        )
```

### **XGBoost Inference Code**
```python
# inference.py (packaged with model artifacts)
import os
import json
import pandas as pd
import xgboost as xgb
from io import StringIO

def model_fn(model_dir):
    """Load XGBoost model from model directory"""
    model_path = os.path.join(model_dir, 'model.xgb')
    model = xgb.Booster()
    model.load_model(model_path)
    return model

def input_fn(request_body, content_type):
    """Parse input data for inference"""
    if content_type == 'text/csv':
        return pd.read_csv(StringIO(request_body))
    elif content_type == 'application/json':
        data = json.loads(request_body)
        return pd.DataFrame(data)
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """Run inference using XGBoost model"""
    dmatrix = xgb.DMatrix(input_data)
    predictions = model.predict(dmatrix)
    return predictions

def output_fn(prediction, accept):
    """Format prediction output"""
    if accept == 'application/json':
        return json.dumps(prediction.tolist())
    elif accept == 'text/csv':
        return ','.join(map(str, prediction))
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
```

## Conclusion

CreateModel step alignment validation patterns provide comprehensive validation for model deployment preparation workflows in SageMaker. The 4-level validation framework is uniquely adapted for CreateModel steps, focusing on:

**Key Differences from Processing/Training Steps:**
- **Builder-focused validation** instead of script-focused validation
- **Container and deployment configuration** validation
- **Model artifact structure** validation
- **Inference code patterns** validation

**Unique Validation Aspects:**
- Level 1: Builder configuration (replaces script validation)
- Level 2: Container deployment configuration
- Level 3: Model artifact structure validation
- Level 4: Model creation builder patterns

This validation pattern ensures that CreateModel steps properly prepare trained models for deployment, handle inference code correctly, and integrate seamlessly with SageMaker's model deployment infrastructure while maintaining framework-specific best practices.

---
tags:
  - design
  - alignment_validation
  - utility_step
  - sagemaker_integration
keywords:
  - utility step validation
  - file preparation patterns
  - configuration generation patterns
  - hyperparameter preparation validation
  - utility script validation
topics:
  - utility step alignment validation
  - utility processing patterns
  - SageMaker utility validation
language: python
date of note: 2025-08-13
---

# Utility Step Alignment Validation Patterns

## Related Documents

### Core Step Type Classification and Patterns
- **[SageMaker Step Type Classification Design](sagemaker_step_type_classification_design.md)** - Complete step type taxonomy and classification system

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
- **[CreateModel Step Alignment Validation Patterns](createmodel_step_alignment_validation_patterns.md)** - CreateModel step validation patterns
- **[Transform Step Alignment Validation Patterns](transform_step_alignment_validation_patterns.md)** - Transform step validation patterns
- **[RegisterModel Step Alignment Validation Patterns](registermodel_step_alignment_validation_patterns.md)** - RegisterModel step validation patterns

## Overview

Utility steps in SageMaker are designed for auxiliary tasks such as file preparation, configuration generation, hyperparameter setup, and other supporting operations. This document defines the specific alignment validation patterns for Utility steps, which serve specialized support functions rather than core ML operations.

## Utility Step Characteristics

### **Core Purpose**
- **File Preparation**: Prepare configuration files and parameters
- **Data Organization**: Organize and structure data for downstream steps
- **Configuration Generation**: Generate hyperparameter and configuration files
- **Support Operations**: Perform auxiliary tasks for ML pipelines

### **SageMaker Integration**
- **Step Type**: `ProcessingStep` (with utility-specific configuration)
- **Processor Types**: `ScriptProcessor`, `SKLearnProcessor` (for utility scripts)
- **Input Types**: Various inputs depending on utility function
- **Output Types**: Configuration files, prepared parameters, organized data

### **Key Characteristics**
- **Special Case Handling**: Often requires custom validation logic
- **Flexible Patterns**: More varied patterns than standard ML steps
- **Support Function**: Enables other steps rather than performing core ML tasks
- **Configuration-Heavy**: Focus on file and parameter preparation

## 4-Level Validation Framework for Utility Steps

### **Level 1: Script Contract Alignment**
Utility scripts must align with their specific utility function contracts.

#### **Required Script Patterns**
```python
# File preparation patterns
import json
import os
import yaml
from pathlib import Path

def prepare_hyperparameters():
    """Prepare hyperparameter configuration files"""
    hyperparams = {
        "max_depth": 6,
        "eta": 0.3,
        "objective": "binary:logistic",
        "num_round": 100
    }
    
    # Save as JSON for SageMaker
    with open('/opt/ml/processing/output/hyperparameters.json', 'w') as f:
        json.dump(hyperparams, f)
    
    # Save as YAML for human readability
    with open('/opt/ml/processing/output/hyperparameters.yaml', 'w') as f:
        yaml.dump(hyperparams, f)

def organize_data_files():
    """Organize data files for downstream processing"""
    input_dir = '/opt/ml/processing/input'
    output_dir = '/opt/ml/processing/output'
    
    # Create organized directory structure
    Path(f"{output_dir}/training").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/validation").mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/test").mkdir(parents=True, exist_ok=True)
    
    # Move and organize files
    for file_path in Path(input_dir).glob("*.csv"):
        if "train" in file_path.name:
            file_path.rename(f"{output_dir}/training/{file_path.name}")
        elif "val" in file_path.name:
            file_path.rename(f"{output_dir}/validation/{file_path.name}")

def generate_configuration():
    """Generate configuration files for pipeline steps"""
    config = {
        "pipeline_config": {
            "instance_type": "ml.m5.large",
            "instance_count": 1,
            "volume_size": 30
        },
        "data_config": {
            "input_format": "csv",
            "output_format": "parquet",
            "compression": "gzip"
        }
    }
    
    with open('/opt/ml/processing/output/pipeline_config.json', 'w') as f:
        json.dump(config, f, indent=2)
```

#### **Environment Variable Usage**
```python
# Utility-specific environment variables
input_path = os.environ.get('SM_CHANNEL_INPUT', '/opt/ml/processing/input')
output_path = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/processing/output')
config_type = os.environ.get('UTILITY_TYPE', 'hyperparameter_prep')
```

#### **Validation Checks**
- ✅ File preparation logic implementation
- ✅ Configuration generation patterns
- ✅ Output file creation and organization
- ✅ Environment variable usage
- ✅ Error handling for file operations

### **Level 2: Contract-Specification Alignment**
Utility contracts must align with step specifications for their specific utility function.

#### **Contract Requirements**
```python
UTILITY_CONTRACT = {
    "inputs": {
        "source_data": {
            "type": "ProcessingInput",
            "source": "s3://bucket/source-data/",
            "destination": "/opt/ml/processing/input"
        },
        "configuration_templates": {
            "type": "ProcessingInput",
            "source": "s3://bucket/templates/",
            "destination": "/opt/ml/processing/templates"
        }
    },
    "outputs": {
        "prepared_files": {
            "type": "ProcessingOutput",
            "source": "/opt/ml/processing/output",
            "destination": "s3://bucket/prepared/"
        },
        "configuration_files": {
            "type": "ProcessingOutput",
            "source": "/opt/ml/processing/config",
            "destination": "s3://bucket/config/"
        }
    },
    "utility_function": "hyperparameter_preparation",
    "environment_variables": {
        "UTILITY_TYPE": "hyperparameter_prep",
        "CONFIG_FORMAT": "json"
    }
}
```

#### **Specification Alignment**
```python
UTILITY_SPEC = {
    "step_name": "hyperparameter-preparation",
    "processor_config": {
        "instance_type": "ml.t3.medium",  # Smaller instance for utility tasks
        "instance_count": 1,
        "volume_size_in_gb": 10
    },
    "inputs": ["ProcessingInput"],
    "outputs": ["ProcessingOutput"],
    "utility_type": "file_preparation",
    "code_location": "s3://bucket/utility-scripts/"
}
```

#### **Validation Checks**
- ✅ Input types match utility function requirements
- ✅ Output types match expected utility outputs
- ✅ Utility function is properly specified
- ✅ Processor configuration is appropriate for utility tasks
- ✅ Environment variables support utility function
- ✅ Code location contains utility scripts

### **Level 3: Specification-Dependency Alignment**
Utility specifications must align with their support role in the pipeline.

#### **Dependency Patterns**
```python
# Utility dependencies (often minimal)
dependencies = {
    "upstream_steps": ["data-ingestion"],  # Minimal upstream dependencies
    "input_artifacts": ["raw_configuration", "template_files"],
    "required_permissions": ["s3:GetObject", "s3:PutObject"],
    "downstream_consumers": ["training-step", "processing-step", "evaluation-step"]  # Many consumers
}
```

#### **Support Function Validation**
```python
# Utility support flow
utility_support_flow = {
    "preparation_phase": "early_pipeline_stage",
    "output_consumers": ["training_step", "processing_step", "evaluation_step"],
    "configuration_targets": ["hyperparameters", "instance_configs", "data_formats"],
    "file_organization": ["training_data", "validation_data", "test_data"]
}

# Configuration generation flow
config_generation_flow = {
    "template_source": "s3://bucket/templates/",
    "parameter_source": "pipeline_parameters",
    "output_configs": ["hyperparameters.json", "pipeline_config.yaml"],
    "target_steps": ["all_downstream_steps"]
}
```

#### **Validation Checks**
- ✅ Utility function supports downstream steps
- ✅ Configuration outputs are consumed by target steps
- ✅ File organization meets pipeline requirements
- ✅ Template and parameter sources are accessible
- ✅ Output formats match consumer expectations
- ✅ Permission requirements are minimal and appropriate

### **Level 4: Builder-Configuration Alignment**
Utility step builders must align with their specific utility function requirements.

#### **Builder Pattern Requirements**
```python
class UtilityStepBuilder:
    def __init__(self):
        self.utility_type = None
        self.processor = None
        self.inputs = []
        self.outputs = []
        self.script_location = None
    
    def _create_processor(self):
        """Create processor for utility tasks (typically lightweight)"""
        return ScriptProcessor(
            image_uri="python:3.8-slim",
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            role=self.role,
            command=["python3"]
        )
    
    def _prepare_files(self):
        """Prepare files for utility function"""
        return [
            ProcessingInput(
                source=self.input_data_uri,
                destination="/opt/ml/processing/input"
            )
        ]
    
    def _configure_outputs(self):
        """Configure utility outputs"""
        return [
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=self.output_data_uri
            )
        ]
    
    def create_step(self):
        """Create utility processing step"""
        return ProcessingStep(
            name=self.step_name,
            processor=self._create_processor(),
            inputs=self._prepare_files(),
            outputs=self._configure_outputs(),
            code=self.script_location
        )
```

#### **Configuration Validation**
```python
# Required builder methods
required_methods = [
    "_create_processor",
    "_prepare_files",
    "_configure_outputs",
    "create_step"
]

# Required configuration parameters
required_config = {
    "utility_type": "hyperparameter_prep",
    "instance_type": "ml.t3.medium",  # Lightweight for utility tasks
    "instance_count": 1,
    "role": "arn:aws:iam::account:role/SageMakerRole",
    "input_data_uri": "s3://bucket/input/",
    "output_data_uri": "s3://bucket/output/",
    "script_location": "s3://bucket/utility-scripts/"
}
```

#### **Validation Checks**
- ✅ Builder implements required methods
- ✅ Processor configuration is lightweight and appropriate
- ✅ Input/output configurations match utility function
- ✅ Script location contains utility scripts
- ✅ Utility type is properly specified
- ✅ Resource configuration is cost-effective

## Utility Function Types

### **Hyperparameter Preparation**
```python
# Hyperparameter preparation utility
def prepare_hyperparameters():
    """Prepare hyperparameters for training steps"""
    base_params = {
        "max_depth": 6,
        "eta": 0.3,
        "objective": "binary:logistic",
        "num_round": 100
    }
    
    # Environment-specific adjustments
    env = os.environ.get('ENVIRONMENT', 'dev')
    if env == 'prod':
        base_params['num_round'] = 200
        base_params['eta'] = 0.1
    
    # Save for SageMaker training
    with open('/opt/ml/processing/output/hyperparameters.json', 'w') as f:
        json.dump(base_params, f)
    
    return base_params

# Builder for hyperparameter preparation
class HyperparameterPrepStepBuilder(UtilityStepBuilder):
    def __init__(self):
        super().__init__()
        self.utility_type = "hyperparameter_prep"
        self.step_name = "hyperparameter-preparation"
```

### **Data Organization**
```python
# Data organization utility
def organize_data_files():
    """Organize data files for ML pipeline"""
    input_dir = '/opt/ml/processing/input'
    output_dir = '/opt/ml/processing/output'
    
    # Create directory structure
    directories = ['training', 'validation', 'test', 'config']
    for dir_name in directories:
        Path(f"{output_dir}/{dir_name}").mkdir(parents=True, exist_ok=True)
    
    # Organize files by type and purpose
    for file_path in Path(input_dir).rglob("*"):
        if file_path.is_file():
            if "train" in file_path.name.lower():
                shutil.copy2(file_path, f"{output_dir}/training/")
            elif "val" in file_path.name.lower():
                shutil.copy2(file_path, f"{output_dir}/validation/")
            elif "test" in file_path.name.lower():
                shutil.copy2(file_path, f"{output_dir}/test/")
            elif file_path.suffix in ['.json', '.yaml', '.yml']:
                shutil.copy2(file_path, f"{output_dir}/config/")

# Builder for data organization
class DataOrganizationStepBuilder(UtilityStepBuilder):
    def __init__(self):
        super().__init__()
        self.utility_type = "data_organization"
        self.step_name = "data-organization"
```

### **Configuration Generation**
```python
# Configuration generation utility
def generate_pipeline_config():
    """Generate configuration for pipeline steps"""
    config = {
        "training_config": {
            "instance_type": os.environ.get('TRAINING_INSTANCE_TYPE', 'ml.m5.xlarge'),
            "instance_count": int(os.environ.get('TRAINING_INSTANCE_COUNT', '1')),
            "max_run": int(os.environ.get('MAX_TRAINING_TIME', '3600'))
        },
        "processing_config": {
            "instance_type": os.environ.get('PROCESSING_INSTANCE_TYPE', 'ml.m5.large'),
            "instance_count": int(os.environ.get('PROCESSING_INSTANCE_COUNT', '1')),
            "volume_size": int(os.environ.get('VOLUME_SIZE', '30'))
        },
        "model_config": {
            "framework": os.environ.get('FRAMEWORK', 'xgboost'),
            "framework_version": os.environ.get('FRAMEWORK_VERSION', '1.5-1'),
            "python_version": os.environ.get('PYTHON_VERSION', 'py3')
        }
    }
    
    # Save configuration
    with open('/opt/ml/processing/output/pipeline_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    return config

# Builder for configuration generation
class ConfigGenerationStepBuilder(UtilityStepBuilder):
    def __init__(self):
        super().__init__()
        self.utility_type = "config_generation"
        self.step_name = "config-generation"
```

## Validation Requirements

### **Required Patterns**
```python
UTILITY_VALIDATION_REQUIREMENTS = {
    "script_patterns": {
        "file_preparation": {
            "keywords": ["prepare", "organize", "generate", "create"],
            "severity": "ERROR"
        },
        "configuration_handling": {
            "keywords": ["json.dump", "yaml.dump", "config", "parameters"],
            "severity": "ERROR"
        },
        "file_operations": {
            "keywords": ["open", "write", "mkdir", "copy", "move"],
            "severity": "ERROR"
        },
        "environment_usage": {
            "keywords": ["os.environ", "getenv", "UTILITY_TYPE"],
            "severity": "WARNING"
        }
    },
    "contract_requirements": {
        "inputs": ["ProcessingInput"],
        "outputs": ["ProcessingOutput"],
        "utility_function": ["hyperparameter_prep", "data_organization", "config_generation"],
        "environment_variables": ["UTILITY_TYPE", "CONFIG_FORMAT"]
    },
    "builder_requirements": {
        "methods": ["_create_processor", "_prepare_files", "_configure_outputs"],
        "configuration": ["utility_type", "instance_type", "script_location"]
    }
}
```

### **Common Issues and Recommendations**

#### **Missing File Preparation Logic**
```python
# Issue: No file preparation implementation
# Recommendation: Add file preparation function
def prepare_files():
    # Create output directories
    Path("/opt/ml/processing/output").mkdir(parents=True, exist_ok=True)
    
    # Prepare configuration files
    config = {"key": "value"}
    with open('/opt/ml/processing/output/config.json', 'w') as f:
        json.dump(config, f)
```

#### **Missing Utility Type Specification**
```python
# Issue: Utility type not specified
# Recommendation: Add utility type configuration
class UtilityStepBuilder:
    def __init__(self):
        self.utility_type = "hyperparameter_prep"  # Specify utility function
```

#### **Missing Output Configuration**
```python
# Issue: No output configuration
# Recommendation: Add output configuration
def _configure_outputs(self):
    return [
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=self.output_data_uri
        )
    ]
```

#### **Missing Environment Variable Usage**
```python
# Issue: No environment variable usage
# Recommendation: Use environment variables for configuration
utility_type = os.environ.get('UTILITY_TYPE', 'default')
config_format = os.environ.get('CONFIG_FORMAT', 'json')
```

## Best Practices

### **Resource Optimization**
- Use lightweight instance types (ml.t3.medium, ml.t3.small)
- Minimize processing time and resource usage
- Use efficient file operations and minimal dependencies
- Implement proper cleanup of temporary files

### **Configuration Management**
- Use environment variables for flexible configuration
- Generate configuration files in standard formats (JSON, YAML)
- Validate configuration parameters before generation
- Implement proper error handling for configuration issues

### **File Operations**
- Create proper directory structures for outputs
- Use atomic file operations to prevent corruption
- Implement proper file permissions and access controls
- Validate file formats and content before processing

### **Pipeline Integration**
- Design utility outputs to match downstream step requirements
- Use consistent naming conventions for generated files
- Implement proper dependency management
- Ensure utility outputs are accessible to consumer steps

## Integration with Step Type Enhancement System

### **Utility Step Enhancer**
```python
class UtilityStepEnhancer(BaseStepEnhancer):
    def __init__(self):
        super().__init__("Utility")
        self.reference_examples = [
            "builder_hyperparameter_prep_step.py"
        ]
        self.utility_validators = {
            "hyperparameter_prep": self._validate_hyperparameter_preparation,
            "data_organization": self._validate_data_organization,
            "config_generation": self._validate_config_generation
        }
    
    def enhance_validation(self, existing_results, script_name):
        additional_issues = []
        
        # Level 1: Utility script patterns
        additional_issues.extend(self._validate_utility_script_patterns(script_name))
        
        # Level 2: Utility specifications
        additional_issues.extend(self._validate_utility_specifications(script_name))
        
        # Level 3: Utility dependencies
        additional_issues.extend(self._validate_utility_dependencies(script_name))
        
        # Level 4: Utility builder patterns
        additional_issues.extend(self._validate_utility_builder(script_name))
        
        return self._merge_results(existing_results, additional_issues)
```

### **Utility Type Detection**
```python
def detect_utility_type(script_content: str) -> Optional[str]:
    """Detect utility type from script content"""
    if 'hyperparameters' in script_content.lower() or 'hyperparams' in script_content.lower():
        return 'hyperparameter_prep'
    elif 'organize' in script_content.lower() or 'directory' in script_content.lower():
        return 'data_organization'
    elif 'config' in script_content.lower() or 'configuration' in script_content.lower():
        return 'config_generation'
    return 'general_utility'
```

## Reference Examples

### **Hyperparameter Preparation Step Builder**
```python
# cursus/steps/builders/builder_hyperparameter_prep_step.py
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep

class HyperparameterPrepStepBuilder:
    def __init__(self):
        self.step_name = "hyperparameter-preparation"
        self.utility_type = "hyperparameter_prep"
        self.instance_type = "ml.t3.medium"
        self.instance_count = 1
        self.role = None
        self.input_data_uri = None
        self.output_data_uri = None
        self.script_location = "hyperparameter_prep.py"
    
    def _create_processor(self):
        return ScriptProcessor(
            image_uri="python:3.8-slim",
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            role=self.role,
            command=["python3"]
        )
    
    def _prepare_files(self):
        return [
            ProcessingInput(
                source=self.input_data_uri,
                destination="/opt/ml/processing/input"
            )
        ]
    
    def _configure_outputs(self):
        return [
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=self.output_data_uri
            )
        ]
    
    def create_step(self):
        return ProcessingStep(
            name=self.step_name,
            processor=self._create_processor(),
            inputs=self._prepare_files(),
            outputs=self._configure_outputs(),
            code=self.script_location
        )
```

### **Hyperparameter Preparation Script**
```python
# hyperparameter_prep.py
import json
import os
import yaml
from pathlib import Path

def main():
    # Get utility configuration
    utility_type = os.environ.get('UTILITY_TYPE', 'hyperparameter_prep')
    config_format = os.environ.get('CONFIG_FORMAT', 'json')
    
    # Prepare hyperparameters
    hyperparams = {
        "max_depth": int(os.environ.get('MAX_DEPTH', '6')),
        "eta": float(os.environ.get('ETA', '0.3')),
        "objective": os.environ.get('OBJECTIVE', 'binary:logistic'),
        "num_round": int(os.environ.get('NUM_ROUND', '100'))
    }
    
    # Create output directory
    output_dir = Path('/opt/ml/processing/output')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save hyperparameters
    if config_format.lower() == 'json':
        with open(output_dir / 'hyperparameters.json', 'w') as f:
            json.dump(hyperparams, f, indent=2)
    elif config_format.lower() == 'yaml':
        with open(output_dir / 'hyperparameters.yaml', 'w') as f:
            yaml.dump(hyperparams, f)
    
    print(f"Hyperparameters prepared: {hyperparams}")
    print(f"Saved in {config_format} format to {output_dir}")

if __name__ == "__main__":
    main()
```

## Conclusion

Utility step alignment validation patterns provide comprehensive validation for auxiliary and support operations in SageMaker pipelines. The 4-level validation framework is adapted for Utility steps, focusing on:

**Key Characteristics:**
- **Special case handling** for varied utility functions
- **Flexible validation patterns** for different utility types
- **Support function validation** for pipeline enablement
- **Configuration-heavy validation** for file and parameter preparation

**Unique Validation Aspects:**
- Level 1: Script contract alignment (utility-specific patterns)
- Level 2: Contract-specification alignment (utility function specification)
- Level 3: Specification-dependency alignment (support role validation)
- Level 4: Builder-configuration alignment (lightweight resource configuration)

This validation pattern ensures that Utility steps properly support ML pipeline operations, generate appropriate configuration files, organize data effectively, and integrate seamlessly with downstream steps while maintaining cost-effective resource usage and proper error handling.

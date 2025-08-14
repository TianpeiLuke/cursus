---
tags:
  - design
  - alignment_validation
  - processing_step
  - sagemaker_integration
keywords:
  - processing step validation
  - data transformation patterns
  - environment variable validation
  - processing input/output validation
  - tabular preprocessing
topics:
  - processing step alignment validation
  - data processing patterns
  - SageMaker processing validation
language: python
date of note: 2025-08-13
---

# Processing Step Alignment Validation Patterns

## Related Documents

### Core Step Type Classification and Patterns
- **[SageMaker Step Type Classification Design](sagemaker_step_type_classification_design.md)** - Complete step type taxonomy and classification system
- **[Processing Step Builder Patterns](processing_step_builder_patterns.md)** - Processing step builder design patterns and implementation guidelines

### Step Type-Aware Validation System
- **[SageMaker Step Type-Aware Unified Alignment Tester Design](sagemaker_step_type_aware_unified_alignment_tester_design.md)** - Main step type-aware validation system design

### Level-Specific Alignment Design Documents
- **[Level 1: Script Contract Alignment Design](level1_script_contract_alignment_design.md)** - Script-contract validation patterns and implementation
- **[Level 2: Contract Specification Alignment Design](level2_contract_specification_alignment_design.md)** - Contract-specification validation patterns
- **[Level 3: Specification Dependency Alignment Design](level3_specification_dependency_alignment_design.md)** - Specification-dependency validation patterns
- **[Level 4: Builder Configuration Alignment Design](level4_builder_configuration_alignment_design.md)** - Builder-configuration validation patterns

### Related Step Type Validation Patterns
- **[Training Step Alignment Validation Patterns](training_step_alignment_validation_patterns.md)** - Training step validation patterns
- **[CreateModel Step Alignment Validation Patterns](createmodel_step_alignment_validation_patterns.md)** - CreateModel step validation patterns
- **[Transform Step Alignment Validation Patterns](transform_step_alignment_validation_patterns.md)** - Transform step validation patterns
- **[RegisterModel Step Alignment Validation Patterns](registermodel_step_alignment_validation_patterns.md)** - RegisterModel step validation patterns
- **[Utility Step Alignment Validation Patterns](utility_step_alignment_validation_patterns.md)** - Utility step validation patterns

## Overview

Processing steps in SageMaker are designed for data transformation, preprocessing, and feature engineering tasks. This document defines the specific alignment validation patterns for Processing steps, which form the foundation of the step type-aware validation system.

## Processing Step Characteristics

### **Core Purpose**
- **Data Transformation**: Transform raw data into processed formats
- **Feature Engineering**: Create and modify features for ML training
- **Data Validation**: Validate data quality and consistency
- **Preprocessing**: Prepare data for training or inference

### **SageMaker Integration**
- **Step Type**: `ProcessingStep`
- **Processor Types**: `SKLearnProcessor`, `PySparkProcessor`, `ScriptProcessor`
- **Input Types**: `ProcessingInput` (data sources)
- **Output Types**: `ProcessingOutput` (transformed data)

## 4-Level Validation Framework for Processing Steps

### **Level 1: Script Contract Alignment**
Processing scripts must align with their contracts for data transformation patterns.

#### **Required Script Patterns**
```python
# Data loading patterns
data = pd.read_csv('/opt/ml/processing/input/data.csv')
data = spark.read.parquet('/opt/ml/processing/input/')

# Data transformation patterns
transformed_data = transform_features(data)
processed_data = preprocess_data(data)

# Data saving patterns
data.to_csv('/opt/ml/processing/output/processed_data.csv')
processed_data.write.parquet('/opt/ml/processing/output/')
```

#### **Environment Variable Usage**
```python
# Required environment variables
input_path = os.environ.get('SM_CHANNEL_INPUT', '/opt/ml/processing/input')
output_path = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/processing/output')
```

#### **Validation Checks**
- ✅ Data loading from SageMaker input paths
- ✅ Data transformation logic implementation
- ✅ Data saving to SageMaker output paths
- ✅ Environment variable usage for path configuration
- ✅ Error handling for data processing failures

### **Level 2: Contract-Specification Alignment**
Processing contracts must align with step specifications for input/output definitions.

#### **Contract Requirements**
```python
PROCESSING_CONTRACT = {
    "inputs": {
        "data_source": {
            "type": "ProcessingInput",
            "source": "s3://bucket/input/",
            "destination": "/opt/ml/processing/input"
        }
    },
    "outputs": {
        "processed_data": {
            "type": "ProcessingOutput",
            "source": "/opt/ml/processing/output",
            "destination": "s3://bucket/output/"
        }
    },
    "environment_variables": {
        "PROCESSING_TYPE": "tabular",
        "FEATURE_COLUMNS": "col1,col2,col3"
    }
}
```

#### **Specification Alignment**
```python
PROCESSING_SPEC = {
    "step_name": "tabular-preprocessing",
    "processor_config": {
        "instance_type": "ml.m5.large",
        "instance_count": 1,
        "volume_size_in_gb": 30
    },
    "inputs": ["ProcessingInput"],
    "outputs": ["ProcessingOutput"],
    "code_location": "s3://bucket/code/"
}
```

#### **Validation Checks**
- ✅ Input types match between contract and specification
- ✅ Output types match between contract and specification
- ✅ Environment variables are properly defined
- ✅ Processor configuration is complete
- ✅ Code location is specified

### **Level 3: Specification-Dependency Alignment**
Processing specifications must align with their dependencies and data flow requirements.

#### **Dependency Patterns**
```python
# Data source dependencies
dependencies = {
    "upstream_steps": ["data-ingestion", "data-validation"],
    "input_artifacts": ["raw_data", "validation_results"],
    "required_permissions": ["s3:GetObject", "s3:PutObject"]
}
```

#### **Data Flow Validation**
```python
# Input data flow
input_data_flow = {
    "source": "upstream_step.properties.ProcessingOutputConfig.Outputs['output_name'].S3Output.S3Uri",
    "destination": "/opt/ml/processing/input",
    "format": "csv|parquet|json"
}

# Output data flow
output_data_flow = {
    "source": "/opt/ml/processing/output",
    "destination": "s3://bucket/processed/",
    "consumers": ["training-step", "evaluation-step"]
}
```

#### **Validation Checks**
- ✅ Upstream step dependencies are satisfied
- ✅ Input data sources are available
- ✅ Output destinations are accessible
- ✅ Data format compatibility
- ✅ Permission requirements are met

### **Level 4: Builder-Configuration Alignment**
Processing step builders must align with their configuration requirements.

#### **Builder Pattern Requirements**
```python
class ProcessingStepBuilder:
    def __init__(self):
        self.processor = None
        self.inputs = []
        self.outputs = []
        self.code_location = None
    
    def _create_processor(self):
        """Create processor instance with proper configuration"""
        return SKLearnProcessor(
            framework_version="0.23-1",
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            role=self.role
        )
    
    def _prepare_inputs(self):
        """Prepare processing inputs"""
        return [
            ProcessingInput(
                source=self.input_data_uri,
                destination="/opt/ml/processing/input"
            )
        ]
    
    def _prepare_outputs(self):
        """Prepare processing outputs"""
        return [
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=self.output_data_uri
            )
        ]
    
    def create_step(self):
        """Create processing step"""
        return ProcessingStep(
            name=self.step_name,
            processor=self._create_processor(),
            inputs=self._prepare_inputs(),
            outputs=self._prepare_outputs(),
            code=self.code_location
        )
```

#### **Configuration Validation**
```python
# Required builder methods
required_methods = [
    "_create_processor",
    "_prepare_inputs", 
    "_prepare_outputs",
    "create_step"
]

# Required configuration parameters
required_config = {
    "instance_type": "ml.m5.large",
    "instance_count": 1,
    "role": "arn:aws:iam::account:role/SageMakerRole",
    "input_data_uri": "s3://bucket/input/",
    "output_data_uri": "s3://bucket/output/",
    "code_location": "s3://bucket/code/script.py"
}
```

#### **Validation Checks**
- ✅ Builder implements required methods
- ✅ Processor configuration is complete
- ✅ Input/output configurations are valid
- ✅ Code location is accessible
- ✅ IAM role has required permissions

## Framework-Specific Patterns

### **Pandas/SKLearn Processing**
```python
# Data loading
import pandas as pd
data = pd.read_csv('/opt/ml/processing/input/data.csv')

# Feature engineering
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[feature_columns])

# Data saving
processed_data.to_csv('/opt/ml/processing/output/processed.csv', index=False)
```

### **PySpark Processing**
```python
# Spark session initialization
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("ProcessingJob").getOrCreate()

# Data loading
df = spark.read.parquet('/opt/ml/processing/input/')

# Data transformation
from pyspark.sql.functions import col, when
transformed_df = df.withColumn("new_feature", when(col("existing_feature") > 0, 1).otherwise(0))

# Data saving
transformed_df.write.mode("overwrite").parquet('/opt/ml/processing/output/')
```

### **Custom Processing**
```python
# Custom transformation logic
def custom_transform(data):
    # Business-specific transformation
    return transformed_data

# Environment-aware processing
processing_type = os.environ.get('PROCESSING_TYPE', 'default')
if processing_type == 'tabular':
    result = tabular_processing(data)
elif processing_type == 'text':
    result = text_processing(data)
```

## Validation Requirements

### **Required Patterns**
```python
PROCESSING_VALIDATION_REQUIREMENTS = {
    "script_patterns": {
        "data_loading": {
            "keywords": ["pd.read_", "spark.read", "load_data"],
            "paths": ["/opt/ml/processing/input"],
            "severity": "ERROR"
        },
        "data_transformation": {
            "keywords": ["transform", "preprocess", "feature_engineering"],
            "severity": "WARNING"
        },
        "data_saving": {
            "keywords": ["to_csv", "write", "save_data"],
            "paths": ["/opt/ml/processing/output"],
            "severity": "ERROR"
        },
        "environment_variables": {
            "keywords": ["os.environ", "SM_CHANNEL_", "SM_OUTPUT_DATA_DIR"],
            "severity": "WARNING"
        }
    },
    "contract_requirements": {
        "inputs": ["ProcessingInput"],
        "outputs": ["ProcessingOutput"],
        "environment_variables": ["PROCESSING_TYPE"]
    },
    "builder_requirements": {
        "methods": ["_create_processor", "_prepare_inputs", "_prepare_outputs"],
        "configuration": ["instance_type", "role", "code_location"]
    }
}
```

### **Common Issues and Recommendations**

#### **Missing Data Loading Patterns**
```python
# Issue: No data loading from SageMaker paths
# Recommendation: Add data loading from processing input
data = pd.read_csv('/opt/ml/processing/input/data.csv')
```

#### **Missing Environment Variable Usage**
```python
# Issue: Hardcoded paths instead of environment variables
# Recommendation: Use SageMaker environment variables
input_path = os.environ.get('SM_CHANNEL_INPUT', '/opt/ml/processing/input')
```

#### **Missing Data Saving Patterns**
```python
# Issue: No data saving to output paths
# Recommendation: Save processed data to output directory
processed_data.to_csv('/opt/ml/processing/output/processed.csv')
```

#### **Incomplete Builder Configuration**
```python
# Issue: Missing required builder methods
# Recommendation: Implement all required methods
def _create_processor(self):
    return SKLearnProcessor(...)

def _prepare_inputs(self):
    return [ProcessingInput(...)]
```

## Best Practices

### **Data Handling**
- Use SageMaker environment variables for path configuration
- Implement proper error handling for data loading/saving
- Validate data quality and schema consistency
- Use appropriate data formats (CSV, Parquet, JSON)

### **Resource Management**
- Configure appropriate instance types for data size
- Use distributed processing for large datasets
- Implement memory-efficient data processing
- Monitor resource utilization

### **Code Organization**
- Separate data loading, transformation, and saving logic
- Use configuration files for processing parameters
- Implement logging for debugging and monitoring
- Follow consistent naming conventions

### **Testing and Validation**
- Test with sample data before full processing
- Validate output data quality and format
- Implement data validation checks
- Use unit tests for transformation logic

## Integration with Step Type Enhancement System

### **Processing Step Enhancer**
```python
class ProcessingStepEnhancer(BaseStepEnhancer):
    def __init__(self):
        super().__init__("Processing")
        self.reference_examples = [
            "tabular_preprocessing.py",
            "risk_table_mapping.py",
            "builder_tabular_preprocessing_step.py"
        ]
    
    def enhance_validation(self, existing_results, script_name):
        additional_issues = []
        
        # Level 1: Processing script patterns
        additional_issues.extend(self._validate_processing_script_patterns(script_name))
        
        # Level 2: Processing specifications
        additional_issues.extend(self._validate_processing_specifications(script_name))
        
        # Level 3: Processing dependencies
        additional_issues.extend(self._validate_processing_dependencies(script_name))
        
        # Level 4: Processing builder patterns
        additional_issues.extend(self._validate_processing_builder(script_name))
        
        return self._merge_results(existing_results, additional_issues)
```

### **Framework Detection**
```python
def detect_processing_framework(script_content: str) -> Optional[str]:
    """Detect processing framework from script content"""
    if 'pandas' in script_content or 'sklearn' in script_content:
        return 'pandas_sklearn'
    elif 'pyspark' in script_content or 'SparkSession' in script_content:
        return 'pyspark'
    elif 'dask' in script_content:
        return 'dask'
    return None
```

## Reference Examples

### **Tabular Preprocessing Script**
```python
# cursus/steps/scripts/tabular_preprocessing.py
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def main():
    # Load data using environment variables
    input_path = os.environ.get('SM_CHANNEL_INPUT', '/opt/ml/processing/input')
    output_path = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/processing/output')
    
    # Load data
    data = pd.read_csv(f'{input_path}/data.csv')
    
    # Feature engineering
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data.select_dtypes(include=[np.number]))
    
    # Save processed data
    processed_data = pd.DataFrame(scaled_features, columns=data.select_dtypes(include=[np.number]).columns)
    processed_data.to_csv(f'{output_path}/processed_data.csv', index=False)

if __name__ == "__main__":
    main()
```

### **Processing Step Builder**
```python
# cursus/steps/builders/builder_tabular_preprocessing_step.py
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.workflow.steps import ProcessingStep

class TabularPreprocessingStepBuilder:
    def __init__(self):
        self.step_name = "tabular-preprocessing"
        self.instance_type = "ml.m5.large"
        self.instance_count = 1
        self.role = None
        self.input_data_uri = None
        self.output_data_uri = None
        self.code_location = None
    
    def _create_processor(self):
        return SKLearnProcessor(
            framework_version="0.23-1",
            instance_type=self.instance_type,
            instance_count=self.instance_count,
            role=self.role
        )
    
    def _prepare_inputs(self):
        return [
            ProcessingInput(
                source=self.input_data_uri,
                destination="/opt/ml/processing/input"
            )
        ]
    
    def _prepare_outputs(self):
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
            inputs=self._prepare_inputs(),
            outputs=self._prepare_outputs(),
            code=self.code_location
        )
```

## Conclusion

Processing step alignment validation patterns provide comprehensive validation for data transformation workflows in SageMaker. The 4-level validation framework ensures proper alignment between scripts, contracts, specifications, and builders, while framework-specific patterns enable targeted validation for different processing technologies.

This validation pattern serves as the foundation for the step type-aware validation system and demonstrates how existing processing validation can be enhanced with step type-specific patterns and requirements.

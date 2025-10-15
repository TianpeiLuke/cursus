---
tags:
  - project
  - implementation
  - lightgbm
  - training_step
  - sagemaker_builtin
  - step_builder
  - machine_learning
keywords:
  - lightgbm training step
  - sagemaker built-in algorithm
  - step builder implementation
  - three-tier configuration
  - specification-driven architecture
  - dependency resolution
  - pipeline integration
topics:
  - lightgbm training step implementation
  - sagemaker built-in algorithm integration
  - step builder architecture
  - configuration management
  - pipeline catalog integration
language: python
date of note: 2025-10-14
---

# Implementation Plan: LightGBM Training Step for Cursus Framework

## Executive Summary

This implementation plan provides a detailed roadmap for implementing a LightGBM Training Step for the Cursus framework using SageMaker's JumpStart LightGBM model. The implementation follows established patterns from the XGBoost Training Step while adapting for LightGBM's specific requirements and JumpStart model approach. The project was executed in 6 phases and is now **COMPLETE**, maintaining full compatibility with the existing specification-driven architecture.

### Key Objectives

#### **Primary Objectives**
- **Implement LightGBM Training Step**: Create complete training step using SageMaker built-in LightGBM algorithm
- **Follow Established Patterns**: Maintain consistency with XGBoost implementation architecture
- **Enable Specification-Driven Architecture**: Full integration with dependency resolution and step catalog
- **Ensure Pipeline Integration**: Seamless integration with existing pipeline system and catalog

#### **Secondary Objectives**
- **Optimize for CPU Instances**: LightGBM-specific instance type recommendations and validation
- **Support Advanced Features**: Distributed training, early stopping, categorical features
- **Maintain Code Quality**: Comprehensive testing, validation, and documentation
- **Enable Workspace Awareness**: Support for project-specific LightGBM configurations

### Strategic Impact

- **ML Algorithm Expansion**: Adds second major gradient boosting algorithm to Cursus framework
- **Built-in Algorithm Pattern**: Establishes pattern for future built-in algorithm integrations
- **Performance Optimization**: CPU-optimized training for cost-effective model development
- **Framework Completeness**: Enhances Cursus as comprehensive ML pipeline framework

## Analysis Summary

### **Current State Assessment**

#### **Available Infrastructure** ✅
- **XGBoost Training Step**: Complete reference implementation with specification-driven architecture
- **Framework Infrastructure**: StepBuilderBase, BasePipelineConfig, StepSpecification system
- **SageMaker Integration**: Proven patterns for TrainingStep creation and dependency resolution
- **Built-in Algorithm Support**: SageMaker SDK supports built-in algorithms via generic Estimator class
- **Validation Framework**: Comprehensive testing tools for alignment and builder validation

#### **Implementation Requirements** 🔨
- **LightGBM Hyperparameter Class**: LightGBMModelHyperparameters inheriting from ModelHyperparameters
- **LightGBM Configuration Class**: Three-tier config design following XGBoost patterns (hyperparameters via source_dir)
- **LightGBM Step Builder**: Builder using sagemaker.estimator.Estimator with LightGBM image URI
- **LightGBM Step Specification**: Dependencies and outputs definition following XGBoost patterns
- **Training Script Contract**: Interface definition for LightGBM training script (if needed)
- **Integration & Testing**: Validation and integration with existing pipeline system

### **Technical Architecture Analysis**

#### **SageMaker JumpStart Model Approach**
Based on research into official AWS SageMaker examples, LightGBM is implemented using JumpStart models:

```python
from sagemaker.estimator import Estimator
from sagemaker import image_uris

# Get LightGBM JumpStart model image URI
image_uri = image_uris.retrieve(
    region=self.aws_region,
    framework=None,
    model_id="lightgbm-classification-model",
    model_version="*",  # Latest version
    image_scope="training",
    instance_type=self.config.training_instance_type,
)

# Create generic estimator with LightGBM JumpStart image
estimator = Estimator(
    image_uri=image_uri,
    role=self.role,
    instance_type=self.config.training_instance_type,
    instance_count=self.config.training_instance_count,
    hyperparameters=self.config.get_lightgbm_hyperparameters(),
    # ... other parameters
)
```

#### **Key Differences from XGBoost Framework Approach**

| Aspect | XGBoost (Framework) | LightGBM (JumpStart Model) |
|--------|-------------------|-------------------------------|
| **Estimator Class** | `sagemaker.xgboost.XGBoost` | `sagemaker.estimator.Estimator` |
| **Image URI** | Framework-specific image | `image_uris.retrieve(model_id="lightgbm-classification-model")` |
| **Model Approach** | Framework container | JumpStart pre-built model |
| **Hyperparameters** | Embedded in source directory | Passed as dictionary to estimator |
| **Entry Point** | Custom training script | JumpStart training script |
| **Data Format** | Flexible | JumpStart model specific requirements |
| **Distributed Training** | Built-in support | Dask-based distributed training |

### **Gap Analysis**

#### **Missing Components** (Implementation Required)
1. **LightGBM Configuration Class**: No existing LightGBM-specific configuration
2. **LightGBM Step Builder**: No builder for built-in algorithm approach
3. **LightGBM Step Specification**: No specification for dependency resolution
4. **Registry Integration**: LightGBM not registered in step registry
5. **Pipeline Templates**: No example pipelines using LightGBM

#### **Available Patterns** (Can Be Adapted)
1. **Three-Tier Configuration**: Established pattern from XGBoost and other steps
2. **Specification-Driven Architecture**: Proven dependency resolution system
3. **Step Builder Base Class**: StepBuilderBase provides foundation
4. **Validation Framework**: Comprehensive testing tools available
5. **Pipeline Catalog Integration**: Established patterns for pipeline creation


## Implementation Plan

Following the correct step creation process:
1. Create contract
2. Create step spec  
3. Register the model in cursus/registry/step_names_original
4. Create config
5. Create step builder

### Phase 1: Create LightGBM Training Contract (Week 1, Day 1) ✅ COMPLETE

#### 1.1 LightGBM Training Contract Implementation ✅
**COMPLETED**: Created `src/cursus/steps/contracts/lightgbm_training_contract.py`

**Implementation Details:**
- Contract name: `LIGHTGBM_TRAIN_CONTRACT`
- Base class: `TrainingScriptContract`
- Input paths: `/opt/ml/input/data`, `/opt/ml/code/hyperparams/hyperparameters.json`
- Output paths: `/opt/ml/model`, `/opt/ml/output/data`
- Framework requirements: LightGBM >=3.0.0 + supporting libraries
- Complete XGBoost alignment for seamless integration

**Original Implementation Code:**

```python
"""
LightGBM Training Script Contract

Defines the contract for the LightGBM training script that handles tabular data
training with risk table mapping and numerical imputation.
Aligned with XGBoost training contract for consistency.
"""

from .training_script_contract import TrainingScriptContract

LIGHTGBM_TRAIN_CONTRACT = TrainingScriptContract(
    entry_point="lightgbm_training.py",
    expected_input_paths={
        "input_path": "/opt/ml/input/data",
        "hyperparameters_s3_uri": "/opt/ml/code/hyperparams/hyperparameters.json",
    },
    expected_output_paths={
        "model_output": "/opt/ml/model",
        "evaluation_output": "/opt/ml/output/data",
    },
    expected_arguments={
        # No expected arguments - using standard paths from contract
    },
    required_env_vars=[
        # No strictly required environment variables - script uses hyperparameters.json
    ],
    optional_env_vars={},
    framework_requirements={
        "boto3": ">=1.26.0",
        "lightgbm": ">=3.0.0",
        "scikit-learn": ">=0.23.2,<1.0.0",
        "pandas": ">=1.2.0,<2.0.0",
        "pyarrow": ">=4.0.0,<6.0.0",
        "beautifulsoup4": ">=4.9.3",
        "flask": ">=2.0.0,<3.0.0",
        "pydantic": ">=2.0.0,<3.0.0",
        "typing-extensions": ">=4.2.0",
        "matplotlib": ">=3.0.0",
        "numpy": ">=1.19.0",
    },
    description="""
    LightGBM training script for tabular data classification that:
    1. Loads training, validation, and test datasets from split directories
    2. Applies numerical imputation using mean strategy for missing values
    3. Fits risk tables on categorical features using training data
    4. Transforms all datasets using fitted preprocessing artifacts
    5. Trains LightGBM model with configurable hyperparameters
    6. Supports both binary and multiclass classification
    7. Handles class weights for imbalanced datasets
    8. Evaluates model performance with comprehensive metrics
    9. Saves model artifacts and preprocessing components
    10. Generates prediction files and performance visualizations
    
    Input Structure (aligned with XGBoost):
    - /opt/ml/input/data: Root directory containing train/val/test subdirectories
      - /opt/ml/input/data/train: Training data files (.csv, .parquet, .json)
      - /opt/ml/input/data/val: Validation data files
      - /opt/ml/input/data/test: Test data files
    - /opt/ml/input/data/config/hyperparameters.json: Model configuration (optional)
    
    Output Structure (aligned with XGBoost):
    - /opt/ml/model: Model artifacts directory
      - /opt/ml/model/lightgbm_model.txt: Trained LightGBM model
      - /opt/ml/model/risk_table_map.pkl: Risk table mappings for categorical features
      - /opt/ml/model/impute_dict.pkl: Imputation values for numerical features
      - /opt/ml/model/feature_importance.json: Feature importance scores
      - /opt/ml/model/feature_columns.txt: Ordered feature column names
      - /opt/ml/model/hyperparameters.json: Model hyperparameters
    - /opt/ml/output/data: Evaluation results directory
      - /opt/ml/output/data/val.tar.gz: Validation predictions and metrics
      - /opt/ml/output/data/test.tar.gz: Test predictions and metrics
    
    Contract aligned with step specification:
    - Inputs: input_path (required), hyperparameters_s3_uri (optional)
    - Outputs: model_output (primary), evaluation_output (secondary)
    
    Hyperparameters (via JSON config):
    - Data fields: tab_field_list, cat_field_list, label_name, id_name
    - Model: is_binary, num_classes, class_weights
    - LightGBM: learning_rate, num_leaves, max_depth, feature_fraction, bagging_fraction
    - Training: num_iterations, early_stopping_rounds
    - Risk tables: smooth_factor, count_threshold
    
    Binary Classification:
    - Uses binary objective
    - Supports scale_pos_weight for class imbalance
    - Generates ROC and PR curves
    - Computes AUC-ROC, Average Precision, F1-Score
    
    Multiclass Classification:
    - Uses multiclass objective
    - Supports sample weights for class imbalance
    - Generates per-class and aggregate metrics
    - Computes micro/macro averaged metrics
    
    Risk Table Processing:
    - Fits risk tables on categorical features using target correlation
    - Applies smoothing and count thresholds for robust estimation
    - Transforms categorical values to risk scores
    
    Numerical Imputation:
    - Uses mean imputation strategy for missing numerical values
    - Fits imputation on training data only
    - Applies same imputation to validation and test sets
    """,
)
```

### Phase 2: Create LightGBM Step Specification (Week 1, Day 2) ✅ COMPLETE

#### 2.1 LightGBM Training Specification ✅
**COMPLETED**: Created `src/cursus/steps/specs/lightgbm_training_spec.py`

**Implementation Details:**
- Specification name: `LIGHTGBM_TRAINING_SPEC`
- Step type: `get_spec_step_type("LightGBMTraining")`
- Node type: `NodeType.INTERNAL`
- Contract reference: `_get_lightgbm_train_contract()`
- Dependencies: `input_path` (required), `hyperparameters_s3_uri` (optional)
- Outputs: `model_output`, `evaluation_output`
- Complete XGBoost alignment with LightGBM-specific aliases

**Original Implementation Code:**

```python
"""
LightGBM Training Step Specification.

This module defines the declarative specification for LightGBM training steps,
following the same patterns as XGBoost but adapted for LightGBM algorithm.
"""

from ...core.base.specification_base import (
    StepSpecification,
    DependencySpec,
    OutputSpec,
    DependencyType,
    NodeType,
)
from ...registry.step_names import get_spec_step_type

# Import the contract at runtime to avoid circular imports
def _get_lightgbm_train_contract():
    from ..contracts.lightgbm_training_contract import LIGHTGBM_TRAIN_CONTRACT
    return LIGHTGBM_TRAIN_CONTRACT

# LightGBM Training Step Specification
LIGHTGBM_TRAINING_SPEC = StepSpecification(
    step_type=get_spec_step_type("LightGBMTraining"),
    node_type=NodeType.INTERNAL,
    script_contract=_get_lightgbm_train_contract(),
    dependencies=[
        DependencySpec(
            logical_name="input_path",
            dependency_type=DependencyType.TRAINING_DATA,
            required=True,
            compatible_sources=[
                "TabularPreprocessing",
                "StratifiedSampling",
                "ProcessingStep",
                "DataLoad",
                "RiskTableMapping",
            ],
            semantic_keywords=[
                "data",
                "input",
                "training",
                "dataset",
                "processed",
                "train",
                "tabular",
            ],
            data_type="S3Uri",
            description="Training dataset S3 location with train/val/test subdirectories",
        ),
        DependencySpec(
            logical_name="hyperparameters_s3_uri",
            dependency_type=DependencyType.HYPERPARAMETERS,
            required=False,  # Can be generated internally
            compatible_sources=["HyperparameterPrep", "ProcessingStep"],
            semantic_keywords=[
                "config",
                "params",
                "hyperparameters",
                "settings",
                "hyperparams",
            ],
            data_type="S3Uri",
            description="S3 URI containing hyperparameters configuration file (optional - falls back to source directory)",
        ),
    ],
    outputs=[
        OutputSpec(
            logical_name="model_output",
            output_type=DependencyType.MODEL_ARTIFACTS,
            property_path="properties.ModelArtifacts.S3ModelArtifacts",
            data_type="S3Uri",
            description="Trained LightGBM model artifacts",
            aliases=[
                "ModelOutputPath",
                "ModelArtifacts",
                "model_data",
                "output_path",
                "model_input",
                "lightgbm_model",
            ],
        ),
        OutputSpec(
            logical_name="evaluation_output",
            output_type=DependencyType.PROCESSING_OUTPUT,
            property_path="properties.OutputDataConfig.S3OutputPath",
            data_type="S3Uri",
            description="Model evaluation results and predictions (val.tar.gz, test.tar.gz)",
            aliases=[
                "evaluation_data",
                "eval_data",
                "validation_output",
                "test_output",
                "prediction_results",
            ],
        ),
    ],
)
```

### Phase 3: Register LightGBM in Step Registry (Week 1, Day 3) ✅ COMPLETE

#### 3.1 Update Step Registry ✅
**COMPLETED**: Added LightGBM entry to `src/cursus/registry/step_names_original.py`

**Implementation Details:**
- Registry entry: `"LightGBMTraining"` in STEP_NAMES dictionary
- Config class: `"LightGBMTrainingConfig"`
- Builder name: `"LightGBMTrainingStepBuilder"`
- Spec type: `"LightGBMTraining"`
- SageMaker step type: `"Training"`
- Description: `"LightGBM model training step using built-in algorithm"`
- Automatic mapping generation: All other registries auto-generated from STEP_NAMES

**Registry Entry Added:**
```python
"LightGBMTraining": {
    "config_class": "LightGBMTrainingConfig",
    "builder_step_name": "LightGBMTrainingStepBuilder",
    "spec_type": "LightGBMTraining",
    "sagemaker_step_type": "Training",
    "description": "LightGBM model training step using built-in algorithm",
},
```

### Phase 4: Create LightGBM Hyperparameter Class (Week 1, Day 4) ✅ COMPLETE

#### 4.1 LightGBM Hyperparameter Implementation ✅
**COMPLETED**: Created `src/cursus/steps/hyperparams/hyperparameters_lightgbm.py`

**Implementation Details:**
- Class name: `LightGBMModelHyperparameters`
- Base class: `ModelHyperparameters`
- Three-tier field classification: Essential (Tier 1), System defaults (Tier 2), Derived (Tier 3)
- Essential fields: `num_leaves`, `learning_rate`
- Derived fields: `objective`, `metric` (auto-calculated from `is_binary`)
- Complete XGBoost alignment with LightGBM-specific parameters
- Comprehensive validation for boosting types, multiclass, early stopping

**Original Implementation Code:**

```python
from pydantic import Field, model_validator, PrivateAttr
from typing import List, Optional, Dict, Any, Union

from ...core.base.hyperparameters_base import ModelHyperparameters


class LightGBMModelHyperparameters(ModelHyperparameters):
    """
    Hyperparameters for the LightGBM model training, extending the base ModelHyperparameters.

    Fields are organized into three tiers:
    1. Tier 1: Essential User Inputs - fields that users must explicitly provide
    2. Tier 2: System Inputs with Defaults - fields with reasonable defaults that can be overridden
    3. Tier 3: Derived Fields - fields calculated from other fields (private attributes with properties)
    """

    # ===== Essential User Inputs (Tier 1) =====
    # These are fields that users must explicitly provide

    # Most essential LightGBM hyperparameters
    num_leaves: int = Field(description="Maximum number of leaves in one tree.")
    
    learning_rate: float = Field(description="Learning rate for boosting.")

    # ===== System Inputs with Defaults (Tier 2) =====
    # These are fields with reasonable defaults that users can override

    # Override model_class from base
    model_class: str = Field(
        default="lightgbm", description="Model class identifier, set to LightGBM."
    )

    # Core LightGBM Parameters
    boosting_type: str = Field(
        default="gbdt",
        description="Boosting type: gbdt, rf, dart, goss.",
    )

    num_iterations: int = Field(
        default=100,
        ge=1,
        description="Number of boosting iterations.",
    )

    max_depth: int = Field(
        default=-1,
        description="Maximum depth of tree. -1 means no limit.",
    )

    min_data_in_leaf: int = Field(
        default=20,
        ge=1,
        description="Minimum number of data points in one leaf.",
    )

    min_sum_hessian_in_leaf: float = Field(
        default=1e-3,
        ge=0.0,
        description="Minimum sum of hessians in one leaf.",
    )

    # Feature Selection Parameters
    feature_fraction: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Feature fraction for each iteration.",
    )

    bagging_fraction: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description="Bagging fraction for each iteration.",
    )

    bagging_freq: int = Field(
        default=0,
        ge=0,
        description="Frequency for bagging. 0 means disable bagging.",
    )

    # Regularization Parameters
    lambda_l1: float = Field(
        default=0.0,
        ge=0.0,
        description="L1 regularization term on weights.",
    )

    lambda_l2: float = Field(
        default=0.0,
        ge=0.0,
        description="L2 regularization term on weights.",
    )

    min_gain_to_split: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum gain to perform split.",
    )

    # Advanced Parameters
    categorical_feature: Optional[str] = Field(
        default=None,
        description="Categorical features specification.",
    )

    early_stopping_rounds: Optional[int] = Field(
        default=None,
        ge=1,
        description="Early stopping rounds. None to disable.",
    )

    seed: Optional[int] = Field(
        default=None, 
        description="Random seed for reproducibility."
    )

    # ===== Derived Fields (Tier 3) =====
    # These are fields calculated from other fields

    _objective: Optional[str] = PrivateAttr(default=None)
    _metric: Optional[Union[str, List[str]]] = PrivateAttr(default=None)

    model_config = ModelHyperparameters.model_config.copy()
    model_config.update({"extra": "allow"})

    # Public read-only properties for derived fields

    @property
    def objective(self) -> str:
        """Get objective derived from is_binary."""
        if self._objective is None:
            self._objective = "binary" if self.is_binary else "multiclass"
        return self._objective

    @property
    def metric(self) -> List[str]:
        """Get evaluation metrics derived from is_binary."""
        if self._metric is None:
            self._metric = (
                ["binary_logloss", "auc"] if self.is_binary else ["multi_logloss", "multi_error"]
            )
        return self._metric

    @model_validator(mode="after")
    def validate_lightgbm_hyperparameters(self) -> "LightGBMModelHyperparameters":
        """Validate LightGBM-specific hyperparameters"""
        # Call the base model validator first
        super().validate_dimensions()

        # Initialize derived fields
        self._objective = "binary" if self.is_binary else "multiclass"
        self._metric = (
            ["binary_logloss", "auc"] if self.is_binary else ["multi_logloss", "multi_error"]
        )

        # Validate multiclass parameters
        if self._objective == "multiclass" and self.num_classes < 2:
            raise ValueError(
                f"For multiclass objective '{self._objective}', 'num_classes' must be >= 2. "
                f"Current num_classes: {self.num_classes}"
            )

        # Validate early stopping configuration
        if self.early_stopping_rounds is not None and not self._metric:
            raise ValueError(
                "'early_stopping_rounds' requires 'metric' to be set."
            )

        # Validate boosting type
        valid_boosting_types = ["gbdt", "rf", "dart", "goss"]
        if self.boosting_type not in valid_boosting_types:
            raise ValueError(
                f"Invalid boosting_type: {self.boosting_type}. Must be one of: {valid_boosting_types}"
            )

        return self

    def get_public_init_fields(self) -> Dict[str, Any]:
        """
        Override get_public_init_fields to include LightGBM-specific fields.
        Gets a dictionary of public fields suitable for initializing a child config.
        """
        # Get fields from parent class
        base_fields = super().get_public_init_fields()

        # Add derived fields that should be exposed
        derived_fields = {"objective": self.objective, "metric": self.metric}

        # Combine (derived fields take precedence if overlap)
        return {**base_fields, **derived_fields}
```

### Phase 5: Create LightGBM Configuration Class (Week 1, Day 5) ✅ COMPLETE

#### 5.1 LightGBM Configuration Implementation ✅
**COMPLETED**: Created `src/cursus/steps/configs/config_lightgbm_training_step.py`

**Implementation Details:**
- Class name: `LightGBMTrainingConfig`
- Base class: `BasePipelineConfig`
- Three-tier field classification: Essential (Tier 1), System defaults (Tier 2), Derived (Tier 3)
- Essential fields: `training_entry_point`
- System defaults: `training_instance_type`, `training_instance_count`, `training_volume_size`, `model_class`, `skip_hyperparameters_s3_uri`
- Derived fields: `hyperparameter_file` (auto-calculated from pipeline location)
- CPU-optimized instance validation for LightGBM built-in algorithm
- Complete XGBoost alignment with LightGBM-specific optimizations

**Original Implementation Code:**

```python
from pydantic import Field, field_validator
from typing import Optional

from ...core.base.config_base import BasePipelineConfig


class LightGBMTrainingConfig(BasePipelineConfig):
    """
    Configuration for LightGBM Training Step using three-tier field classification.
    
    Tier 1: Essential fields (required user inputs)
    Tier 2: System fields (with defaults, can be overridden)  
    Tier 3: Derived fields (private with property access)
    
    Hyperparameters are managed by the user - they must create a LightGBMModelHyperparameters
    instance and save it as hyperparameters.json in the source_dir. The training script
    will load this JSON file from the container's source directory.
    """
    
    # ===== Essential User Inputs (Tier 1) =====
    training_entry_point: str = Field(
        description="Entry point script for LightGBM training."
    )
    
    # ===== System Inputs with Defaults (Tier 2) =====
    training_instance_type: str = Field(
        default="ml.m5.4xlarge", 
        description="Instance type for LightGBM training job."
    )
    training_instance_count: int = Field(
        default=1, ge=1, 
        description="Number of instances for LightGBM training job."
    )
    training_volume_size: int = Field(
        default=30, ge=1, 
        description="Volume size (GB) for training instances."
    )
    
    # Override model_class to match hyperparameters
    model_class: str = Field(
        default="lightgbm", 
        description="Model class identifier, set to LightGBM."
    )
    
    @field_validator("training_instance_type")
    @classmethod
    def validate_lightgbm_instance_type(cls, v: str) -> str:
        """Validate instance types suitable for LightGBM."""
        # LightGBM works well on CPU instances
        valid_cpu_instances = [
            "ml.m5.large", "ml.m5.xlarge", "ml.m5.2xlarge", "ml.m5.4xlarge",
            "ml.c5.large", "ml.c5.xlarge", "ml.c5.2xlarge", "ml.c5.4xlarge",
            "ml.r5.large", "ml.r5.xlarge", "ml.r5.2xlarge"  # Memory-optimized for large datasets
        ]
        if v not in valid_cpu_instances:
            raise ValueError(f"Invalid instance type for LightGBM: {v}")
        return v
```

### Phase 6: Create LightGBM Step Builder (Week 1, Day 6) ✅ COMPLETE

#### 6.1 LightGBM Step Builder Implementation ✅
**COMPLETED**: Created `src/cursus/steps/builders/builder_lightgbm_training_step.py`

**Implementation Details:**
- Class name: `LightGBMTrainingStepBuilder`
- Base class: `StepBuilderBase`
- JumpStart model approach: Uses `sagemaker.estimator.Estimator` with LightGBM JumpStart model image URI
- Image URI retrieval: `image_uris.retrieve(model_id="lightgbm-classification-model", ...)`
- Specification integration: Uses `LIGHTGBM_TRAINING_SPEC` for dependency resolution
- Complete XGBoost alignment with JumpStart model adaptations
- Environment variables: Minimal environment setup (no redundant variables)

**Actual Implementation Code:**

```python
def _get_lightgbm_image_uri(self) -> str:
    """Get LightGBM JumpStart model image URI for the current region."""
    # LightGBM is available as a JumpStart model, not a traditional built-in algorithm
    return image_uris.retrieve(
        region=self.aws_region,
        framework=None,
        model_id="lightgbm-classification-model",
        model_version="*",  # Latest version
        image_scope="training",
        instance_type=self.config.training_instance_type,
    )

def _create_estimator(self, output_path=None) -> Estimator:
    """
    Creates and configures the LightGBM estimator using JumpStart model.
    This defines the execution environment for the training job, including the instance
    type, JumpStart model image, and environment variables.
    """
    # Get LightGBM JumpStart model image URI
    image_uri = self._get_lightgbm_image_uri()
    self.log_info("Using LightGBM JumpStart model image: %s", image_uri)

    # Use modernized effective_source_dir with comprehensive hybrid resolution
    source_dir = self.config.effective_source_dir
    self.log_info("Using source directory: %s", source_dir)
    
    return Estimator(
        image_uri=image_uri,
        entry_point=self.config.training_entry_point,
        source_dir=source_dir,
        role=self.role,
        instance_type=self.config.training_instance_type,
        instance_count=self.config.training_instance_count,
        volume_size=self.config.training_volume_size,
        max_run=86400,  # 24 hours default
        input_mode='File',
        output_path=output_path,  # Use provided output_path directly
        base_job_name=self._generate_job_name(),  # Use standardized method with auto-detection
        sagemaker_session=self.session,
        environment=self._get_environment_variables(),
    )

def _get_environment_variables(self) -> Dict[str, str]:
    """
    Constructs a dictionary of environment variables to be passed to the training job.
    For JumpStart models, minimal environment variables are needed.
    """
    # Get base environment variables from contract
    env_vars = super()._get_environment_variables()

    # Add environment variables from config if they exist
    if hasattr(self.config, "env") and self.config.env:
        env_vars.update(self.config.env)

    self.log_info("Training environment variables: %s", env_vars)
    return env_vars
```

**Key Implementation Features:**
- **JumpStart Model Integration**: Uses official SageMaker JumpStart LightGBM model
- **Verified Image URI**: Based on official AWS SageMaker examples
- **Clean Environment Setup**: No redundant environment variables (contract specifies none required)
- **Complete XGBoost Alignment**: Same patterns with JumpStart-specific adaptations
- **Distributed Training Ready**: Supports multi-instance distributed training with Dask

### Phase 7: Integration and Testing (Week 2, Days 1-3)

#### 7.1 Validation Framework Testing
Run comprehensive validation tests:

```bash
# Test step specification alignment
cursus validate-alignment --step LightGBMTraining --workspace main --verbose

# Test step builder functionality
cursus validate-builder --step LightGBMTraining --workspace main --verbose

# Test script runtime (if custom script is used)
cursus runtime test-script lightgbm_training --workspace-dir ./test_workspace --verbose
```

#### 7.2 Unit Testing Implementation
Create comprehensive unit tests:

```python
# test/steps/builders/test_builder_lightgbm_training_step.py
class TestLightGBMTrainingStepBuilder(unittest.TestCase):
    def setUp(self):
        self.config = LightGBMTrainingConfig(
            region="us-west-2",
            pipeline_s3_loc="s3://bucket/prefix",
            training_entry_point="lightgbm_training.py",
            objective="regression",
            num_leaves=31,
            learning_rate=0.1
        )
        self.builder = LightGBMTrainingStepBuilder(self.config)
    
    def test_initialization(self):
        """Test that the builder initializes correctly with specification."""
        self.assertIsNotNone(self.builder.spec)
        self.assertEqual(self.builder.spec.step_type, "LightGBMTraining")
        self.assertEqual(self.builder.spec.node_type, NodeType.INTERNAL)
    
    def test_create_estimator(self):
        """Test that LightGBM estimator is created correctly."""
        estimator = self.builder._create_estimator()
        
        # Verify it's using built-in algorithm approach
        self.assertIsInstance(estimator, Estimator)
        self.assertIn("lightgbm", estimator.image_uri)
        
        # Verify hyperparameters are set correctly
        expected_hyperparams = self.config.get_lightgbm_hyperparameters()
        self.assertEqual(estimator.hyperparameters, expected_hyperparams)
    
    def test_get_lightgbm_image_uri(self):
        """Test that LightGBM image URI is retrieved correctly."""
        image_uri = self.builder._get_lightgbm_image_uri()
        
        # Should contain lightgbm in the URI
        self.assertIsNotNone(image_uri)
        self.assertIn("lightgbm", image_uri.lower())
    
    def test_get_inputs(self):
        """Test that inputs are correctly derived from dependencies."""
        inputs = {"input_path": "s3://bucket/input/data"}
        training_inputs = self.builder._get_inputs(inputs)
        
        # Should create train/val/test channels from input_path
        self.assertIn("train", training_inputs)
        self.assertIn("val", training_inputs)
        self.assertIn("test", training_inputs)
    
    def test_get_outputs(self):
        """Test that outputs are correctly configured."""
        output_path = self.builder._get_outputs({})
        
        # Should generate default output path
        self.assertIsNotNone(output_path)
        # Should contain lightgbm_training in the path
        self.assertIn("lightgbm_training", str(output_path))
    
    @patch('cursus.steps.builders.builder_lightgbm_training_step.image_uris.retrieve')
    def test_create_step(self, mock_retrieve):
        """Test step creation with dependencies."""
        # Mock image URI retrieval
        mock_retrieve.return_value = "123456789.dkr.ecr.us-west-2.amazonaws.com/lightgbm:latest"
        
        # Create step
        inputs = {"input_path": "s3://bucket/input/data"}
        step = self.builder.create_step(inputs=inputs)
        
        # Verify step was created
        self.assertIsNotNone(step)
        self.assertIsInstance(step, TrainingStep)
        self.assertTrue(hasattr(step, '_spec'))
        
        # Verify image URI was retrieved for LightGBM
        mock_retrieve.assert_called_with("lightgbm", region=self.builder.aws_region)

if __name__ == '__main__':
    unittest.main()
```

### Phase 8: Advanced Features and Optimization (Week 2, Days 4-7)

#### 8.1 LightGBM-Specific Features
Implement advanced LightGBM features:

```python
class AdvancedLightGBMTrainingConfig(LightGBMTrainingConfig):
    """Extended LightGBM configuration with advanced features."""
    
    # Advanced LightGBM parameters
    boosting_type: str = Field(
        default="gbdt", 
        description="Boosting type (gbdt, rf, dart, goss)"
    )
    num_iterations: int = Field(
        default=100, ge=1, 
        description="Number of boosting iterations"
    )
    early_stopping_rounds: Optional[int] = Field(
        default=None, 
        description="Early stopping rounds (None to disable)"
    )
    categorical_feature: Optional[str] = Field(
        default=None, 
        description="Categorical features specification"
    )
    
    # Distributed training support
    enable_distributed_training: bool = Field(
        default=False, 
        description="Enable distributed training with Dask"
    )
    
    @field_validator("boosting_type")
    @classmethod
    def validate_boosting_type(cls, v: str) -> str:
        """Validate boosting type."""
        valid_types = ["gbdt", "rf", "dart", "goss"]
        if v not in valid_types:
            raise ValueError(f"Invalid boosting type: {v}. Must be one of: {valid_types}")
        return v
    
    def get_advanced_lightgbm_hyperparameters(self) -> Dict[str, str]:
        """Get advanced LightGBM hyperparameters."""
        base_params = super().get_lightgbm_hyperparameters()
        
        advanced_params = {
            "boosting_type": self.boosting_type,
            "num_iterations": str(self.num_iterations),
        }
        
        if self.early_stopping_rounds is not None:
            advanced_params["early_stopping_rounds"] = str(self.early_stopping_rounds)
        
        if self.categorical_feature is not None:
            advanced_params["categorical_feature"] = self.categorical_feature
        
        base_params.update(advanced_params)
        return base_params
```

#### 8.2 Distributed Training Support
Add support for distributed LightGBM training:

```python
def _create_distributed_estimator(self, output_path=None) -> Estimator:
    """Create estimator with distributed training support."""
    if not self.config.enable_distributed_training:
        return self._create_estimator(output_path)
    
    from sagemaker.estimator import Estimator
    from sagemaker import image_uris
    
    # Get LightGBM image with Dask support
    image_uri = image_uris.retrieve("lightgbm", region=self.aws_region, version="latest")
    
    # Configure for distributed training
    distribution = {
        "smdistributed": {
            "dataparallel": {
                "enabled": True
            }
        }
    }
    
    return Estimator(
        image_uri=image_uri,
        role=self.role,
        instance_type=self.config.training_instance_type,
        instance_count=max(2, self.config.training_instance_count),  # Minimum 2 for distributed
        volume_size=self.config.training_volume_size,
        distribution=distribution,
        hyperparameters=self.config.get_advanced_lightgbm_hyperparameters(),
        # ... other parameters
    )
```

#### 8.3 Performance Optimization
Implement performance optimizations:

```python
class OptimizedLightGBMTrainingStepBuilder(LightGBMTrainingStepBuilder):
    """Optimized LightGBM builder with performance enhancements."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._estimator_cache = {}
        self._input_cache = {}
    
    def _create_estimator(self, output_path=None) -> Estimator:
        """Create estimator with caching for repeated builds."""
        cache_key = (
            self.config.training_instance_type,
            self.config.training_instance_count,
            str(self.config.get_lightgbm_hyperparameters()),
            output_path
        )
        
        if cache_key not in self._estimator_cache:
            self._estimator_cache[cache_key] = super()._create_estimator(output_path)
        
        return self._estimator_cache[cache_key]
    
    def _optimize_instance_selection(self) -> str:
        """Automatically select optimal instance type based on data size."""
        # This could analyze input data size and recommend instance types
        # For now, return configured instance type
        return self.config.training_instance_type
```

### Phase 9: Documentation and Examples (Week 3, Days 1-2)

#### 9.1 Usage Documentation
Create comprehensive usage documentation:

```markdown
# LightGBM Training Step Usage Guide

## Basic Usage

```python
from cursus.steps.configs.config_lightgbm_training_step import LightGBMTrainingConfig
from cursus.steps.builders.builder_lightgbm_training_step import LightGBMTrainingStepBuilder

# Create configuration
config = LightGBMTrainingConfig(
    region="us-west-2",
    pipeline_s3_loc="s3://my-bucket/pipeline",
    training_entry_point="lightgbm_training.py",
    objective="binary",
    num_leaves=50,
    learning_rate=0.05,
    feature_fraction=0.8
)

# Create step builder
builder = LightGBMTrainingStepBuilder(config)

# Create step with dependencies
step = builder.create_step(
    inputs={"input_path": "s3://my-bucket/processed-data"},
    dependencies=[preprocessing_step]
)
```

## Advanced Configuration

```python
# Advanced LightGBM configuration
advanced_config = AdvancedLightGBMTrainingConfig(
    region="us-west-2",
    pipeline_s3_loc="s3://my-bucket/pipeline",
    objective="multiclass",
    boosting_type="dart",
    num_iterations=200,
    early_stopping_rounds=10,
    enable_distributed_training=True,
    training_instance_count=4
)
```

## Pipeline Integration

```python
# Use in pipeline
from cursus.pipeline_catalog.pipelines.lightgbm_training_pipeline import create_pipeline

pipeline, report, dag_compiler = create_pipeline(
    config_path="path/to/lightgbm_config.json",
    session=pipeline_session,
    role=role
)
```


#### 9.2 Example Pipeline Implementation
Create example pipeline using LightGBM:

```python
# src/cursus/pipeline_catalog/pipelines/lightgbm_training_pipeline.py
def create_lightgbm_training_dag() -> PipelineDAG:
    """Create a LightGBM training pipeline DAG."""
    dag = PipelineDAG()
    
    # Add nodes
    dag.add_node("CradleDataLoading_training")
    dag.add_node("TabularPreprocessing_training")
    dag.add_node("LightGBMTraining")
    dag.add_node("ModelRegistration")
    
    # Create pipeline flow
    dag.add_edge("CradleDataLoading_training", "TabularPreprocessing_training")
    dag.add_edge("TabularPreprocessing_training", "LightGBMTraining")
    dag.add_edge("LightGBMTraining", "ModelRegistration")
    
    return dag
```

## Implementation Timeline

### Week 1: Core Implementation (5 days)
- **Days 1-2**: LightGBM Configuration Class with validation
- **Days 3-4**: LightGBM Step Builder with built-in algorithm support
- **Days 5-6**: Step Specification and Contract definition
- **Day 7**: Basic integration testing

### Week 2: Testing and Advanced Features (7 days)
- **Days 1-3**: Comprehensive testing and validation framework
- **Days 4-5**: Advanced features (distributed training, optimization)
- **Days 6-7**: Performance testing and benchmarking

### Week 3: Documentation and Polish (5 days)
- **Days 1-2**: Documentation and usage examples
- **Days 3-4**: Pipeline integration and example pipelines
- **Day 5**: Final testing and production readiness

**Total Timeline: 3 weeks**

## Success Criteria

### Functional Requirements ✅
1. **Built-in Algorithm Integration**: Successfully uses SageMaker built-in LightGBM algorithm
2. **Specification-Driven**: Follows same patterns as XGBoost with automatic dependency resolution
3. **Hyperparameter Support**: Comprehensive LightGBM hyperparameter configuration
4. **Data Channel Handling**: Proper train/validation/test data channel creation
5. **Pipeline Integration**: Seamless integration with existing pipeline system

### Performance Requirements ✅
1. **Training Performance**: Comparable or better performance than custom implementations
2. **Resource Efficiency**: Optimal instance type selection and resource utilization
3. **Distributed Training**: Support for multi-instance distributed training
4. **Caching**: Efficient caching for repeated pipeline builds

### Quality Requirements ✅
1. **Test Coverage**: >90% test coverage for all LightGBM components
2. **Validation**: Comprehensive validation using cursus validation framework
3. **Documentation**: Complete documentation with examples and best practices
4. **Error Handling**: Clear error messages and robust error handling

## Risk Mitigation

### Technical Risks
1. **Built-in Algorithm Limitations**: Risk of limited customization compared to framework approach
   - **Mitigation**: Thorough testing of built-in algorithm capabilities and fallback to custom implementation if needed

2. **Hyperparameter Compatibility**: Risk of hyperparameter format mismatches
   - **Mitigation**: Comprehensive validation of hyperparameter formats and SageMaker compatibility

3. **Data Format Requirements**: Risk of built-in algorithm having specific data format requirements
   - **Mitigation**: Research and document data format requirements, implement data validation

### Integration Risks
1. **Specification Alignment**: Risk of specification not matching actual built-in algorithm behavior
   - **Mitigation**: Extensive testing with real training jobs and validation framework

2. **Dependency Resolution**: Risk of dependency resolution issues with existing steps
   - **Mitigation**: Comprehensive testing with existing preprocessing and data loading steps

## Comparison with XGBoost Implementation

### Similarities ✅
- Three-tier configuration design
- Specification-driven architecture
- Dependency resolution patterns
- Validation framework integration
- Pipeline catalog integration

### Key Differences 🔄
- Uses generic `Estimator` instead of framework-specific class
- Hyperparameters passed as dictionary instead of embedded in source
- Built-in algorithm image URI retrieval
- Potentially simpler training script requirements
- Different instance type recommendations (CPU-optimized)

## Next Steps

### Immediate Actions (Week 1)
1. **Create Configuration Class**: Implement `LightGBMTrainingConfig` with comprehensive hyperparameter support
2. **Build Step Builder**: Implement `LightGBMTrainingStepBuilder` using built-in algorithm approach
3. **Define Specification**: Create `LIGHTGBM_TRAINING_SPEC` following XGBoost patterns
4. **Registry Integration**: Add LightGBM to step registry and naming conventions

### Validation Actions (Week 2)
1. **Run Validation Framework**: Test with cursus validation tools
2. **Integration Testing**: Test with existing preprocessing and data loading steps
3. **Performance Benchmarking**: Compare with XGBoost and custom implementations
4. **Advanced Features**: Implement distributed training and optimization features

### Production Readiness (Week 3)
1. **Documentation**: Complete usage guides and API documentation
2. **Example Pipelines**: Create example pipelines demonstrating LightGBM usage
3. **Production Testing**: End-to-end testing with real datasets
4. **Performance Optimization**: Final performance tuning and optimization

## Conclusion

The LightGBM Training Step implementation is ready for immediate development. With the comprehensive analysis of the XGBoost implementation and clear understanding of SageMaker built-in algorithms, this project can be completed efficiently following established patterns.

### Key Success Factors

1. **Proven Architecture**: Following established XGBoost patterns ensures consistency and reliability
2. **Built-in Algorithm Benefits**: Leveraging SageMaker built-in LightGBM provides optimized performance and maintenance
3. **Comprehensive Planning**: Detailed implementation plan reduces development risks
4. **Validation Framework**: Existing validation tools ensure quality and integration
5. **Clear Timeline**: 3-week timeline with clear milestones and deliverables

### Immediate Next Steps

1. **Start with Configuration Class** (Day 1): Foundation for all other components
2. **Implement Step Builder** (Days 2-3): Core functionality using built-in algorithm
3. **Create Specification** (Day 4): Enable automatic dependency resolution
4. **Validate Integration** (Day 5): Ensure compatibility with existing system

The project is ready for immediate implementation with **high confidence of success** and **clear deliverables**. All architectural decisions are based on proven patterns and comprehensive analysis.

## References

### Developer Guide Documentation
- **[Adding New Pipeline Step](../slipbox/0_developer_guide/adding_new_pipeline_step.md)**: Complete guide for adding new pipeline steps to the Cursus framework
- **[Step Builder Guide](../slipbox/0_developer_guide/step_builder.md)**: Detailed documentation on step builder implementation patterns
- **[Step Specification Guide](../slipbox/0_developer_guide/step_specification.md)**: Specification-driven architecture documentation
- **[Hyperparameter Class Guide](../slipbox/0_developer_guide/hyperparameter_class.md)**: Three-tier hyperparameter class design patterns
- **[Config Field Manager Guide](../slipbox/0_developer_guide/config_field_manager_guide.md)**: Configuration management and field categorization
- **[Three-Tier Config Design](../slipbox/0_developer_guide/three_tier_config_design.md)**: Configuration architecture principles
- **[Validation Framework Guide](../slipbox/0_developer_guide/validation_framework_guide.md)**: Testing and validation patterns
- **[Step Catalog Integration Guide](../slipbox/0_developer_guide/step_catalog_integration_guide.md)**: Pipeline catalog integration patterns

### Design Documentation
- **[Config-Driven Design](../slipbox/1_design/config_driven_design.md)**: Configuration-driven architecture principles
- **[Specification-Driven Architecture](../slipbox/1_design/adaptive_specification_integration.md)**: Adaptive specification integration design
- **[Step Builder Patterns](../slipbox/1_design/createmodel_step_builder_patterns.md)**: Step builder implementation patterns
- **[Dependency Resolution System](../slipbox/1_design/dependency_resolution_system.md)**: Automatic dependency resolution design
- **[Config Field Categorization](../slipbox/1_design/config_field_categorization_consolidated.md)**: Field categorization and management
- **[Three-Tier Config Implementation](../slipbox/1_design/config_manager_three_tier_implementation.md)**: Three-tier configuration implementation
- **[Enhanced Property Reference](../slipbox/1_design/enhanced_property_reference.md)**: Property path and reference management

### Related Implementation Examples
- **XGBoost Training Step**: Reference implementation for gradient boosting algorithms
  - `src/cursus/steps/builders/builder_xgboost_training_step.py`
  - `src/cursus/steps/configs/config_xgboost_training_step.py`
  - `src/cursus/steps/hyperparams/hyperparameters_xgboost.py`
- **Step Registry**: `src/cursus/registry/step_names_original.py`
- **Base Classes**: 
  - `src/cursus/core/base/builder_base.py`
  - `src/cursus/core/base/config_base.py`
  - `src/cursus/core/base/hyperparameters_base.py`

### External Documentation
- **[AWS SageMaker LightGBM Examples](https://github.com/aws/amazon-sagemaker-examples/blob/main/introduction_to_applying_machine_learning/sagemaker_lightgbm_distributed_training_dask/sagemaker-lightgbm-distributed-training-dask.ipynb)**: Official AWS SageMaker LightGBM JumpStart model examples
- **[SageMaker SDK Documentation](https://sagemaker.readthedocs.io/en/stable/api/utility/image_uris.html)**: Image URI retrieval documentation
- **[LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/)**: Official LightGBM algorithm documentation

---
tags:
  - code
  - base
  - hyperparameters_base
  - model_hyperparameters
  - three_tier_design
keywords:
  - ModelHyperparameters
  - hyperparameter management
  - three-tier design
  - field categorization
  - model configuration
  - classification parameters
  - derived fields
topics:
  - hyperparameter management
  - model configuration
  - three-tier design
language: python
date of note: 2024-12-07
---

# Hyperparameters Base

Base model hyperparameters for training tasks that implements a three-tier design pattern for organizing hyperparameter fields with automatic derivation and comprehensive validation capabilities.

## Overview

The `ModelHyperparameters` class provides the foundational hyperparameter management system for machine learning training tasks, implementing the same sophisticated three-tier design pattern used in the configuration system. This class organizes hyperparameters by their purpose and derivation logic, ensuring clear separation between user inputs, system defaults, and computed values.

The three-tier design includes Essential User Inputs (Tier 1) that are required fields users must explicitly provide such as field lists and classification parameters, System Inputs with Defaults (Tier 2) that have reasonable defaults but can be overridden including model parameters and training settings, and Derived Fields (Tier 3) that are calculated from other fields such as input dimensions and classification type detection.

The system supports advanced features including automatic field categorization and validation, comprehensive model dimension validation, flexible hyperparameter inheritance patterns, SageMaker-compatible serialization, and detailed hyperparameter analysis and debugging capabilities.

## Classes and Methods

### Classes
- [`ModelHyperparameters`](#modelhyperparameters) - Base model hyperparameters with three-tier design and automatic derivation

## API Reference

### ModelHyperparameters

_class_ cursus.core.base.hyperparameters_base.ModelHyperparameters(_full_field_list_, _cat_field_list_, _tab_field_list_, _id_name_, _label_name_, _multiclass_categories_, _categorical_features_to_encode=[]_, _model_class="base_model"_, _device=-1_, _header=0_, _lr=3e-05_, _batch_size=2_, _max_epochs=3_, _metric_choices=["f1_score", "auroc"]_, _optimizer="SGD"_, _class_weights=None_)

Base model hyperparameters for training tasks. This class implements the three-tier design pattern for organizing hyperparameter fields and provides comprehensive validation and derivation capabilities for machine learning models.

**Parameters:**
- **full_field_list** (_List[str]_) – Full list of original field names. Required essential user input.
- **cat_field_list** (_List[str]_) – Categorical fields using original names. Required essential user input.
- **tab_field_list** (_List[str]_) – Tabular/numeric fields using original names. Required essential user input.
- **id_name** (_str_) – ID field name. Required essential user input.
- **label_name** (_str_) – Label field name. Required essential user input.
- **multiclass_categories** (_List[Union[int, str]]_) – List of unique category labels. Required essential user input.
- **categorical_features_to_encode** (_List[str]_) – List of categorical fields requiring specific encoding. Defaults to empty list.
- **model_class** (_str_) – Model class name. Defaults to "base_model".
- **device** (_int_) – Device ID for training (-1 for CPU). Defaults to -1.
- **header** (_int_) – Header row for CSV files. Defaults to 0.
- **lr** (_float_) – Learning rate. Defaults to 3e-05.
- **batch_size** (_int_) – Batch size for training (1-256). Defaults to 2.
- **max_epochs** (_int_) – Maximum epochs for training (1-10). Defaults to 3.
- **metric_choices** (_List[str]_) – Metric choices for evaluation. Defaults to ["f1_score", "auroc"].
- **optimizer** (_str_) – Optimizer type. Defaults to "SGD".
- **class_weights** (_Optional[List[float]]_) – Class weights for loss function. Defaults to [1.0] * num_classes.

```python
from cursus.core.base.hyperparameters_base import ModelHyperparameters

# Create hyperparameters for binary classification
hyperparams = ModelHyperparameters(
    full_field_list=["feature1", "feature2", "feature3", "id", "label"],
    cat_field_list=["feature1"],
    tab_field_list=["feature2", "feature3"],
    id_name="id",
    label_name="label",
    multiclass_categories=["positive", "negative"],
    model_class="xgboost",
    lr=0.01,
    batch_size=32,
    max_epochs=10,
    optimizer="Adam"
)

print(f"Input dimension: {hyperparams.input_tab_dim}")
print(f"Number of classes: {hyperparams.num_classes}")
print(f"Is binary classification: {hyperparams.is_binary}")
```

### Properties (Derived Fields - Tier 3)

#### input_tab_dim

_property_ input_tab_dim

Get input tabular dimension derived from tab_field_list. This derived property calculates the number of tabular/numeric features for model input layer sizing.

**Returns:**
- **int** – Number of tabular fields, calculated as len(tab_field_list).

```python
# Access derived input dimension
print(f"Model input dimension: {hyperparams.input_tab_dim}")
# Automatically calculated from tab_field_list length
```

#### num_classes

_property_ num_classes

Get number of classes derived from multiclass_categories. This derived property determines the output layer size for classification models.

**Returns:**
- **int** – Number of classes, calculated as len(multiclass_categories).

```python
# Access derived number of classes
print(f"Output classes: {hyperparams.num_classes}")
# Automatically calculated from multiclass_categories length
```

#### is_binary

_property_ is_binary

Determine if this is a binary classification task based on num_classes. This derived property enables conditional logic for binary vs multiclass scenarios.

**Returns:**
- **bool** – True if num_classes == 2, False otherwise.

```python
# Check classification type
if hyperparams.is_binary:
    print("Using binary classification metrics")
else:
    print("Using multiclass classification metrics")
```

### Methods

#### categorize_fields

categorize_fields()

Categorize all fields into three tiers based on their characteristics and purpose. This method provides insight into the hyperparameter structure and field organization.

**Returns:**
- **Dict[str, List[str]]** – Dictionary with keys 'essential', 'system', and 'derived' mapping to lists of field names.

```python
# Analyze field categorization
categories = hyperparams.categorize_fields()

print("Essential fields:", categories['essential'])
print("System fields:", categories['system'])
print("Derived fields:", categories['derived'])

# Output:
# Essential fields: ['full_field_list', 'cat_field_list', 'tab_field_list', 'id_name', 'label_name', 'multiclass_categories']
# System fields: ['categorical_features_to_encode', 'model_class', 'device', 'header', 'lr', 'batch_size', 'max_epochs', 'metric_choices', 'optimizer', 'class_weights']
# Derived fields: ['input_tab_dim', 'num_classes', 'is_binary']
```

#### print_hyperparam

print_hyperparam()

Print complete hyperparameter information organized by tiers. This method provides a comprehensive view of the hyperparameters with fields organized by their tier classification.

```python
# Print organized hyperparameters
hyperparams.print_hyperparam()

# Output:
# ===== HYPERPARAMETERS =====
# Class: ModelHyperparameters
# 
# ----- Essential User Inputs (Tier 1) -----
# Full_Field_List: ['feature1', 'feature2', 'feature3', 'id', 'label']
# Cat_Field_List: ['feature1']
# ...
# 
# ----- System Inputs with Defaults (Tier 2) -----
# Model_Class: xgboost
# Lr: 0.01
# ...
# 
# ----- Derived Fields (Tier 3) -----
# Input_Tab_Dim: 2
# Num_Classes: 2
# Is_Binary: True
```

#### get_public_init_fields

get_public_init_fields()

Get a dictionary of public fields suitable for initializing a child hyperparameter. This method extracts all user-provided and system fields that should be propagated to derived hyperparameter classes.

**Returns:**
- **Dict[str, Any]** – Dictionary of field names to values for child initialization.

```python
# Get fields for child hyperparameter
init_fields = hyperparams.get_public_init_fields()
print("Fields for child hyperparams:", list(init_fields.keys()))

# Use for creating derived hyperparameter
child_hyperparams = DerivedHyperparameters(**init_fields, additional_param="value")
```

#### get_config

get_config()

Get the complete configuration dictionary. This method returns the full hyperparameter configuration as a dictionary suitable for serialization and storage.

**Returns:**
- **Dict[str, Any]** – Complete configuration dictionary with all hyperparameter values.

```python
# Get complete configuration
config = hyperparams.get_config()
print("Configuration keys:", list(config.keys()))
```

#### serialize_config

serialize_config()

Serialize configuration for SageMaker. This method converts all hyperparameter values to string format suitable for SageMaker training job parameters.

**Returns:**
- **Dict[str, str]** – Serialized configuration with all values as strings, including derived fields.

```python
# Serialize for SageMaker
serialized = hyperparams.serialize_config()
print("Serialized config:")
for key, value in serialized.items():
    print(f"  {key}: {value} (type: {type(value).__name__})")

# All values are strings suitable for SageMaker
```

### Class Methods

#### from_base_hyperparam

_classmethod_ from_base_hyperparam(_base_hyperparam_, _**kwargs_)

Create a new hyperparameter instance from a base hyperparameter. This method enables hyperparameter inheritance and specialization patterns.

**Parameters:**
- **base_hyperparam** (_ModelHyperparameters_) – Parent ModelHyperparameters instance.
- ****kwargs** – Additional arguments specific to the derived class.

**Returns:**
- **ModelHyperparameters** – New instance of the derived class initialized with parent fields and additional kwargs.

```python
# Create derived hyperparameters from base
class XGBoostHyperparameters(ModelHyperparameters):
    n_estimators: int = Field(default=100, description="Number of trees")
    max_depth: int = Field(default=6, description="Maximum tree depth")

# Inherit from base hyperparameters
xgb_hyperparams = XGBoostHyperparameters.from_base_hyperparam(
    hyperparams,
    n_estimators=200,
    max_depth=8
)

print(f"Inherited field lists: {xgb_hyperparams.full_field_list}")
print(f"XGBoost n_estimators: {xgb_hyperparams.n_estimators}")
print(f"Derived input_tab_dim: {xgb_hyperparams.input_tab_dim}")
```

## Usage Examples

### Basic Hyperparameter Creation
```python
from cursus.core.base.hyperparameters_base import ModelHyperparameters

# Create hyperparameters for multiclass classification
hyperparams = ModelHyperparameters(
    full_field_list=["age", "income", "education", "category", "id", "target"],
    cat_field_list=["education", "category"],
    tab_field_list=["age", "income"],
    id_name="id",
    label_name="target",
    multiclass_categories=["low", "medium", "high"],
    categorical_features_to_encode=["education"],
    model_class="neural_network",
    lr=0.001,
    batch_size=64,
    max_epochs=20,
    optimizer="Adam",
    class_weights=[1.0, 2.0, 1.5]  # Handle class imbalance
)

# Access derived properties
print(f"Input dimension: {hyperparams.input_tab_dim}")  # 2 (age, income)
print(f"Number of classes: {hyperparams.num_classes}")  # 3 (low, medium, high)
print(f"Is binary: {hyperparams.is_binary}")  # False
```

### Hyperparameter Analysis and Debugging
```python
# Analyze hyperparameter structure
def analyze_hyperparams(hyperparams):
    """Analyze hyperparameter field organization."""
    categories = hyperparams.categorize_fields()
    
    print(f"Hyperparameter Analysis for {hyperparams.__class__.__name__}:")
    print(f"  Essential fields: {len(categories['essential'])}")
    print(f"  System fields: {len(categories['system'])}")
    print(f"  Derived fields: {len(categories['derived'])}")
    
    # Show field details
    for category, fields in categories.items():
        print(f"\n{category.title()} Fields:")
        for field in sorted(fields):
            try:
                value = getattr(hyperparams, field)
                if isinstance(value, list) and len(value) > 3:
                    print(f"  {field}: [{len(value)} items]")
                else:
                    print(f"  {field}: {value}")
            except Exception as e:
                print(f"  {field}: <Error: {e}>")

# Analyze the hyperparameters
analyze_hyperparams(hyperparams)

# Print organized view
hyperparams.print_hyperparam()
```

### Hyperparameter Inheritance Pattern
```python
# Create specialized hyperparameter class
class DeepLearningHyperparameters(ModelHyperparameters):
    """Deep learning specific hyperparameters."""
    
    # Additional deep learning fields
    hidden_layers: List[int] = Field(default=[128, 64], description="Hidden layer sizes")
    dropout_rate: float = Field(default=0.2, description="Dropout rate")
    activation: str = Field(default="relu", description="Activation function")
    
    # Override derived field if needed
    @property
    def total_parameters(self) -> int:
        """Estimate total model parameters."""
        params = self.input_tab_dim * self.hidden_layers[0]
        for i in range(1, len(self.hidden_layers)):
            params += self.hidden_layers[i-1] * self.hidden_layers[i]
        params += self.hidden_layers[-1] * self.num_classes
        return params

# Create deep learning hyperparams from base
dl_hyperparams = DeepLearningHyperparameters.from_base_hyperparam(
    hyperparams,
    hidden_layers=[256, 128, 64],
    dropout_rate=0.3,
    activation="gelu"
)

print(f"Hidden layers: {dl_hyperparams.hidden_layers}")
print(f"Estimated parameters: {dl_hyperparams.total_parameters}")

# Verify inheritance
assert dl_hyperparams.full_field_list == hyperparams.full_field_list
assert dl_hyperparams.input_tab_dim == hyperparams.input_tab_dim
```

### SageMaker Integration
```python
# Serialize hyperparameters for SageMaker training job
serialized_config = hyperparams.serialize_config()

print("SageMaker hyperparameters:")
for key, value in sorted(serialized_config.items()):
    print(f"  '{key}': '{value}'")

# Use in SageMaker estimator
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    entry_point="train.py",
    role="arn:aws:iam::123456789012:role/SageMakerRole",
    instance_type="ml.m5.large",
    instance_count=1,
    framework_version="1.12.0",
    py_version="py39",
    hyperparameters=serialized_config
)

# In training script, deserialize hyperparameters
import json
import argparse

def parse_hyperparameters():
    parser = argparse.ArgumentParser()
    
    # Add all hyperparameter arguments
    for key, value in serialized_config.items():
        parser.add_argument(f"--{key}", type=str, default=value)
    
    args = parser.parse_args()
    
    # Deserialize complex types
    hyperparams_dict = {}
    for key, value in vars(args).items():
        try:
            # Try to parse as JSON for lists/dicts/bools
            hyperparams_dict[key] = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            # Keep as string for simple types
            hyperparams_dict[key] = value
    
    return hyperparams_dict

# Recreate hyperparameters in training script
training_hyperparams = parse_hyperparameters()
```

### Validation and Error Handling
```python
# Test hyperparameter validation
try:
    # Invalid batch size (too large)
    invalid_hyperparams = ModelHyperparameters(
        full_field_list=["f1", "f2", "id", "label"],
        cat_field_list=["f1"],
        tab_field_list=["f2"],
        id_name="id",
        label_name="label",
        multiclass_categories=["a", "b"],
        batch_size=500  # Exceeds maximum of 256
    )
except ValueError as e:
    print(f"Validation error: {e}")

try:
    # Mismatched class weights
    invalid_hyperparams = ModelHyperparameters(
        full_field_list=["f1", "f2", "id", "label"],
        cat_field_list=["f1"],
        tab_field_list=["f2"],
        id_name="id",
        label_name="label",
        multiclass_categories=["a", "b", "c"],  # 3 classes
        class_weights=[1.0, 2.0]  # Only 2 weights
    )
except ValueError as e:
    print(f"Class weights validation error: {e}")
```

## Related Documentation

- [Config Base](config_base.md) - Base configuration system with similar three-tier design
- [Specification Base](specification_base.md) - Step specifications that may reference hyperparameters
- [Builder Base](builder_base.md) - Step builders that use hyperparameter configurations
- [Base Enums](enums.md) - Enumerations used in hyperparameter validation
- [Contract Base](contract_base.md) - Script contracts that may specify hyperparameter requirements

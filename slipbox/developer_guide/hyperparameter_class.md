# Adding a New Hyperparameter Class

When adding a new training step, you will likely need to create a custom hyperparameter class that inherits from the base `ModelHyperparameters` class. This guide walks you through this process.

## Overview

Hyperparameter classes define model-specific configuration parameters that control the behavior of training algorithms. By using a dedicated class for hyperparameters:

- Configuration is type-safe and self-documenting
- Default values are centralized
- Validation rules ensure parameter values are appropriate
- The pipeline can properly serialize and deserialize configurations

## Implementation Steps

### Step 1: Create the Hyperparameter Class

1. Create a new file in the `src/pipeline_steps` directory following the naming convention `hyperparameters_<model_type>.py`:

```python
# src/pipeline_steps/hyperparameters_mymodel.py

from typing import Dict, Any, Optional
from pydantic import Field, validator

from .hyperparameters_base import ModelHyperparameters

class MyModelHyperparameters(ModelHyperparameters):
    """
    Hyperparameters for MyModel training.
    
    This class defines all configurable parameters specific to MyModel training.
    """
    learning_rate: float = Field(0.01, description="Learning rate for the optimizer")
    num_epochs: int = Field(10, description="Number of training epochs")
    batch_size: int = Field(32, description="Mini-batch size for training")
    
    # Model-specific hyperparameters
    hidden_layers: int = Field(2, description="Number of hidden layers in the network")
    hidden_units: int = Field(128, description="Number of units per hidden layer")
    dropout_rate: float = Field(0.5, description="Dropout rate for regularization")
    
    # Add validators if needed
    @validator('learning_rate')
    def validate_learning_rate(cls, v):
        if v <= 0:
            raise ValueError("Learning rate must be positive")
        return v
```

### Step 2: Register the Hyperparameter Class

Register your new hyperparameter class in the central registry at `src/pipeline_registry/hyperparameter_registry.py`:

```python
# Update the HYPERPARAMETER_REGISTRY dictionary

HYPERPARAMETER_REGISTRY = {
    # ... existing entries ...
    
    "MyModelHyperparameters": {
        "class_name": "MyModelHyperparameters",
        "module_path": "src.pipeline_steps.hyperparameters_mymodel",
        "model_type": "mymodel",
        "description": "Hyperparameters for MyModel models"
    }
}
```

### Step 3: Update the Training Config Class

In your training step configuration class, reference your hyperparameter class:

```python
# src/pipeline_steps/config_mymodeltraining.py

from typing import Dict, Any, Optional
from pydantic import Field

from .config_base import BasePipelineConfig
from .hyperparameters_mymodel import MyModelHyperparameters

class MyModelTrainingConfig(BasePipelineConfig):
    """Configuration for MyModel training step."""
    
    # Basic step configuration
    job_type: str = "training"
    instance_type: str = "ml.m5.xlarge"
    instance_count: int = 1
    
    # Model-specific hyperparameters
    hyperparameters: MyModelHyperparameters = Field(
        default_factory=MyModelHyperparameters,
        description="Hyperparameters for MyModel training"
    )
    
    # Other training configuration...
```

### Step 4: Integrate with the Training Script

Ensure your training script properly uses the hyperparameters:

```python
# dockers/mymodel/train.py

import json
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    # Add hyperparameters as arguments
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--num-epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-layers', type=int, default=2)
    parser.add_argument('--hidden-units', type=int, default=128)
    parser.add_argument('--dropout-rate', type=float, default=0.5)
    # Other arguments...
    return parser.parse_args()

def train():
    args = parse_args()
    # Use args.learning_rate, args.num_epochs, etc.
    # ...
```

### Step 5: Ensure Step Builder Passes Hyperparameters

Verify that your step builder correctly passes the hyperparameters to the SageMaker training job:

```python
# src/pipeline_builder/mymodel_training_step_builder.py

def build_step(self, config, specification):
    # ...
    
    # Convert hyperparameters to the format expected by SageMaker
    hyperparameters = {
        'learning_rate': str(config.hyperparameters.learning_rate),
        'num_epochs': str(config.hyperparameters.num_epochs),
        'batch_size': str(config.hyperparameters.batch_size),
        'hidden_layers': str(config.hyperparameters.hidden_layers),
        'hidden_units': str(config.hyperparameters.hidden_units),
        'dropout_rate': str(config.hyperparameters.dropout_rate),
    }
    
    # Create SageMaker training job step
    step = TrainingStep(
        # ...
        hyperparameters=hyperparameters,
        # ...
    )
    
    # ...
```

## Benefits of Using the Registry System

Using the hyperparameter registry system provides several advantages:

1. **Single Source of Truth**: All hyperparameter classes are registered in one place
2. **Dynamic Loading**: Classes are loaded dynamically when needed
3. **Metadata Storage**: Additional information about each class is stored with its registration
4. **Discoverability**: Developers can easily find existing hyperparameter classes
5. **Consistency**: Encourages consistent naming and organization
6. **Extensibility**: Makes it easy to add new hyperparameter classes without modifying core code

## Testing

Remember to run the tests after adding your hyperparameter class to ensure everything works correctly:

```bash
python -m unittest test/pipeline_steps/test_utils_refactor.py
```

## Related Resources

- [Design Principles](design_principles.md)
- [Config Field Categorization](../pipeline_design/config_field_categorization.md)
- [Config Field Manager Refactoring](../pipeline_design/config_field_manager_refactoring.md)

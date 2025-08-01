# PyTorch Training Script Contract

## Overview
The PyTorch Training Script Contract defines the execution requirements for the PyTorch Lightning multimodal training script (`train.py`). This contract ensures the script complies with SageMaker training job conventions and framework requirements.

## Contract Details

### Script Information
- **Entry Point**: `train.py`
- **Container Type**: SageMaker Training Job
- **Framework**: PyTorch Lightning
- **Model Types**: Multimodal (text + tabular), BERT, CNN, LSTM variants

### Input Path Requirements

| Logical Name | Expected Path | Description |
|--------------|---------------|-------------|
| train_data | `/opt/ml/input/data/train` | Training dataset files |
| val_data | `/opt/ml/input/data/val` | Validation dataset files |
| test_data | `/opt/ml/input/data/test` | Test dataset files |
| config | `/opt/ml/input/config/hyperparameters.json` | Model configuration and hyperparameters |

### Output Path Requirements

| Logical Name | Expected Path | Description |
|--------------|---------------|-------------|
| model_output | `/opt/ml/model` | Trained model artifacts |
| data_output | `/opt/ml/output/data` | Training outputs and metrics |
| checkpoints | `/opt/ml/checkpoints` | Training checkpoints |

### Environment Variables

#### Required Variables
- None (script uses hyperparameters.json for configuration)

#### Optional Variables
| Variable | Default | Description |
|----------|---------|-------------|
| SM_CHECKPOINT_DIR | `/opt/ml/checkpoints` | SageMaker checkpoint directory |

### Framework Requirements

#### Core PyTorch Stack
```python
torch==2.1.2
torchvision==0.16.2
torchaudio==2.1.2
lightning==2.1.3
lightning-utilities==0.10.1
torchmetrics==1.7.1
```

#### NLP and Transformers
```python
transformers==4.37.2
gensim==4.3.1
beautifulsoup4==4.12.3
```

#### Data Processing
```python
pandas==2.1.4
pyarrow==14.0.2
scikit-learn==1.3.2
```

#### Visualization and Monitoring
```python
tensorboard==2.16.2
matplotlib==3.8.2
```

#### Model Export
```python
onnx==1.15.0
onnxruntime==1.17.0
```

#### Validation and Serving
```python
pydantic==2.11.2
flask==3.0.2
```

## Script Functionality

### Data Processing Pipeline
1. **Data Loading** - Loads multimodal data (text + tabular features)
2. **Text Preprocessing** - Tokenization, chunking, and encoding
3. **Categorical Processing** - Feature encoding and label processing
4. **Data Splitting** - Train/validation/test split handling

### Model Architecture Support
- **multimodal_bert** - BERT + tabular fusion
- **multimodal_cnn** - CNN + tabular fusion  
- **multimodal_moe** - Mixture of Experts multimodal
- **multimodal_gate_fusion** - Gated fusion multimodal
- **multimodal_cross_attn** - Cross-attention multimodal
- **bert** - Text-only BERT classification
- **lstm** - Text-only LSTM classification

### Training Features
- **PyTorch Lightning** - Distributed training support
- **Early Stopping** - Configurable patience and monitoring
- **Checkpointing** - Automatic checkpoint saving
- **Mixed Precision** - FP16 training support
- **Gradient Clipping** - Configurable gradient clipping

### Output Artifacts
- **model.pth** - Trained PyTorch model
- **model_artifacts.pth** - Model configuration and embeddings
- **model.onnx** - ONNX exported model
- **predict_results.pth** - Prediction results
- **tensorboard_eval/** - TensorBoard evaluation logs

## Hyperparameter Configuration

### Required Parameters
```json
{
  "model_class": "multimodal_bert",
  "id_name": "id",
  "text_name": "text_field",
  "label_name": "target",
  "num_classes": 2,
  "is_binary": true
}
```

### Data Configuration
```json
{
  "tab_field_list": ["feature1", "feature2", "feature3"],
  "cat_field_list": ["category1", "category2"],
  "tokenizer": "bert-base-uncased",
  "max_sen_len": 512,
  "max_total_chunks": 10
}
```

### Training Configuration
```json
{
  "batch_size": 32,
  "max_epochs": 10,
  "lr": 2e-5,
  "optimizer": "adamw",
  "early_stop_patience": 3,
  "warmup_steps": 1000
}
```

### Model Configuration
```json
{
  "hidden_common_dim": 256,
  "class_weights": [1.0, 2.0],
  "fp16": true,
  "gradient_clip_val": 1.0,
  "reinit_layers": 2
}
```

## Validation Results

### Current Compliance Status
```bash
train.py vs TrainingScriptContract: ❌ NON-COMPLIANT
Errors: [
  "Script doesn't use expected input path: /opt/ml/input/data/train (for train_data)",
  "Script doesn't use expected input path: /opt/ml/input/data/val (for val_data)"
]
```

### Non-Compliance Explanation
The script uses dynamic path construction rather than hardcoded strings:
```python
# Script uses variables and path joining
prefix = "/opt/ml/"
input_path = os.path.join(prefix, "input/data")
train_path = os.path.join(input_path, "train")
```

This is expected behavior and indicates robust path handling rather than contract violation.

## Usage Example

### Contract Access
```python
from src.pipeline_script_contracts import PYTORCH_TRAIN_CONTRACT

# Access contract details
print(f"Entry Point: {PYTORCH_TRAIN_CONTRACT.entry_point}")
print(f"Framework Requirements: {PYTORCH_TRAIN_CONTRACT.framework_requirements}")
print(f"Expected Inputs: {PYTORCH_TRAIN_CONTRACT.expected_input_paths}")
```

### Validation
```python
from src.pipeline_script_contracts import ScriptContractValidator

# Validate script in PyTorch container directory
validator = ScriptContractValidator('dockers/pytorch_bsm')
report = validator.validate_script('train.py')
print(report.summary)
```

### Integration with Step Builder
```python
from src.pipeline_steps import PyTorchTrainingStepBuilder

class PyTorchTrainingStepBuilder(StepBuilderBase):
    def validate_configuration(self) -> None:
        # Validate script compliance
        validation = PYTORCH_TRAIN_CONTRACT.validate_implementation(
            'dockers/pytorch_bsm/train.py'
        )
        if not validation.is_valid:
            self.logger.warning(f"Script validation warnings: {validation.errors}")
```

## Integration Points

### With Step Specifications
```python
from src.pipeline_step_specs import PYTORCH_TRAINING_SPEC

# Contract aligns with step specification
assert PYTORCH_TRAIN_CONTRACT.entry_point == "train.py"
assert PYTORCH_TRAINING_SPEC.step_name == "pytorch_training"
```

### With Pipeline Builder
```python
# Contract validates script before pipeline execution
pipeline_builder.validate_script_contracts()
```

## Best Practices

### Script Development
1. **Use Contract Paths** - Reference contract-defined paths in documentation
2. **Environment Variables** - Handle optional environment variables gracefully
3. **Framework Versions** - Pin exact versions as specified in contract
4. **Error Handling** - Implement robust error handling for missing inputs

### Contract Maintenance
1. **Version Alignment** - Keep framework versions aligned with requirements.txt
2. **Path Validation** - Ensure paths match SageMaker conventions
3. **Documentation Updates** - Update contract when script functionality changes
4. **Testing** - Validate contract compliance in CI/CD pipelines

## Related Contracts

### Training Contracts
- `XGBOOST_TRAIN_CONTRACT` - XGBoost training script contract
- `PYTORCH_INFERENCE_CONTRACT` - PyTorch inference script contract

### Processing Contracts
- `TABULAR_PREPROCESS_CONTRACT` - Preprocessing for tabular data
- `MODEL_EVALUATION_CONTRACT` - Model evaluation script contract

## Troubleshooting

### Common Issues
1. **Path Not Found** - Ensure input data is mounted correctly
2. **Framework Conflicts** - Verify exact framework versions
3. **Memory Issues** - Adjust batch size and model parameters
4. **Checkpoint Errors** - Ensure checkpoint directory is writable

### Validation Failures
1. **Dynamic Paths** - Expected for robust scripts using path construction
2. **Missing Imports** - AST analysis may not detect all import patterns
3. **Environment Variables** - Optional variables may not be detected in static analysis

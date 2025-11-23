---
tags:
  - code
  - training_script
  - pytorch_lightning
  - multimodal_learning
  - deep_learning
keywords:
  - PyTorch training
  - Lightning framework
  - multimodal models
  - BERT classification
  - text and tabular fusion
  - distributed training
  - FSDP
  - ONNX export
  - early stopping
  - model checkpointing
topics:
  - PyTorch Lightning training
  - multimodal machine learning
  - deep learning workflows
  - model training and evaluation
language: python
date of note: 2025-11-18
---

# PyTorch Training Script

## Overview

The `pytorch_training.py` script implements a comprehensive PyTorch Lightning-based training framework for multimodal Business Seller Messaging (BSM) models that combine text and tabular data for classification tasks.

The script provides a production-ready training pipeline with support for multiple model architectures (BERT, CNN, LSTM, and various multimodal fusion strategies), distributed training with FSDP, automatic mixed precision, early stopping, checkpointing, and model export to multiple formats. It handles the complete workflow from data loading and preprocessing through model training, evaluation, and artifact generation.

Key capabilities include:
- **Bimodal and Trimodal Support**: Single text field (bimodal) or dual text fields with independent processing pipelines (trimodal)
- **Risk-Based Categorical Encoding**: Risk table mapping for categorical features instead of simple label encoding
- **Numerical Imputation**: Single-column architecture for numerical feature imputation with artifact saving
- **Format Preservation**: Automatic detection and preservation of input data format (CSV/TSV/Parquet) for predictions
- **Pre-computed Artifact Support**: Optional loading of pre-computed imputation and risk tables from upstream steps
- **Comprehensive Model Export**: PyTorch (.pth), ONNX (.onnx), and preprocessing artifacts (.pkl) for deployment
- **Distributed Training**: FSDP and DDP support with synchronization barriers for multi-GPU training
- **Flexible Text Processing**: Configurable processing steps per text field for bimodal and trimodal architectures

## Purpose and Major Tasks

### Primary Purpose
Train multimodal deep learning models that combine text (dialogue/messaging) and tabular (numerical/categorical) features for classification tasks, with support for multiple architectures and production deployment requirements.

### Major Tasks

1. **Hyperparameter Loading and Validation**: Load and validate training configuration from JSON with Pydantic-based schema validation
2. **Data Loading**: Load train/validation/test datasets from multiple file formats (CSV, TSV, Parquet)
3. **Text Preprocessing Pipeline**: Apply dialogue splitting, HTML normalization, emoji removal, text normalization, chunking, and BERT tokenization
4. **Categorical Feature Encoding**: Build and apply categorical label processors for tabular features
5. **Multiclass Label Processing**: Handle multiclass classification with custom label mappings and validation
6. **Model Architecture Selection**: Instantiate appropriate model from 7+ supported architectures (BERT, CNN, LSTM, multimodal variants)
7. **Training with Lightning**: Train using PyTorch Lightning with distributed training support, early stopping, and checkpointing
8. **Model Evaluation**: Compute comprehensive metrics (AUROC, F1, Average Precision) on validation and test sets
9. **Visualization Generation**: Create ROC curves, precision-recall curves, and TensorBoard logs
10. **Model Export**: Save trained models in PyTorch format and export to ONNX for deployment

## Script Contract

### Entry Point
```
pytorch_training.py
```

### Input Paths

| Path | Location | Description |
|------|----------|-------------|
| `input_path` | `/opt/ml/input/data` | Root directory containing train/val/test subdirectories |
| `train` | `/opt/ml/input/data/train` | Training data files (.csv, .tsv, .parquet) |
| `val` | `/opt/ml/input/data/val` | Validation data files |
| `test` | `/opt/ml/input/data/test` | Test data files |
| `hyperparameters_s3_uri` | `/opt/ml/code/hyperparams/hyperparameters.json` | Model configuration and hyperparameters |
| `model_artifacts_input` | `/opt/ml/input/data/model_artifacts_input` | Optional: Pre-computed preprocessing artifacts |

**Optional Preprocessing Artifacts** (in model_artifacts_input):
- `impute_dict.pkl`: Pre-computed imputation parameters
- `risk_table_map.pkl`: Pre-computed risk tables
- `selected_features.json`: Pre-computed feature selection

### Output Paths

| Path | Location | Description |
|------|----------|-------------|
| `model_output` | `/opt/ml/model` | Primary model artifacts directory |
| `evaluation_output` | `/opt/ml/output/data` | Evaluation results and metrics |
| `checkpoints` | `/opt/ml/checkpoints` | Training checkpoints (intermediate) |

**Model Output Contents**:
- `model.pth`: Trained PyTorch model (state dict)
- `model_artifacts.pth`: Model artifacts (config, embeddings, vocabulary)
- `model.onnx`: ONNX exported model for deployment
- `impute_dict.pkl`: Numerical imputation parameters (for inference)
- `impute_dict.json`: Human-readable imputation parameters
- `risk_table_map.pkl`: Risk table mappings (for inference)
- `risk_table_map.json`: Human-readable risk tables

**Evaluation Output Contents**:
- `predict_results.pth`: Prediction results (true labels, predicted probabilities)
- `tensorboard_eval/`: TensorBoard evaluation logs with metrics and plots
- `failure`: Failure file with error details (only on error)

### Required Environment Variables

None strictly required - all configuration via hyperparameters.json

### Optional Environment Variables

#### Preprocessing Artifact Control
| Variable | Default | Description | Use Case |
|----------|---------|-------------|----------|
| `USE_PRECOMPUTED_IMPUTATION` | `"false"` | Use pre-computed imputation artifacts from model_artifacts_input | When numerical imputation done upstream (e.g., in separate imputation step) |
| `USE_PRECOMPUTED_RISK_TABLES` | `"false"` | Use pre-computed risk table artifacts from model_artifacts_input | When risk table mapping done upstream (e.g., in separate risk table step) |

**Note**: When these flags are true, the script loads pre-computed artifacts from `/opt/ml/input/data/model_artifacts_input/` instead of fitting them inline during training. This enables decoupled preprocessing workflows.

### Job Arguments

**No command-line arguments** - Script follows container contract pattern with fixed paths and configuration via hyperparameters.json

### Hyperparameters (via JSON Configuration)

#### Data Configuration
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `id_name` | `str` | Yes | Column name for record IDs | `"order_id"` |
| `text_name` | `str` | Conditional | Column name for text data (bimodal only, mutually exclusive with primary/secondary_text_name) | `"text"` |
| `primary_text_name` | `str` | Conditional | Primary text field name (trimodal only, e.g., chat messages) | `"chat"` |
| `secondary_text_name` | `str` | Conditional | Secondary text field name (trimodal only, e.g., event logs) | `"shiptrack"` |
| `label_name` | `str` | Yes | Column name for classification labels | `"label"` |
| `full_field_list` | `List[str]` | Yes | Complete list of all data fields | `["order_id", "text", "label", ...]` |
| `cat_field_list` | `List[str]` | Yes | Categorical feature column names (risk table mapping applied) | `["marketplace", "category"]` |
| `tab_field_list` | `List[str]` | Yes | Numerical feature column names (imputation applied) | `["price", "quantity", "rating"]` |
| `train_filename` | `str` | No | Explicit train file name (auto-detected if omitted) | `"train.csv"` |
| `val_filename` | `str` | No | Explicit validation file name | `"val.csv"` |
| `test_filename` | `str` | No | Explicit test file name | `"test.csv"` |

**Note on Text Field Configuration**:
- **Bimodal**: Use `text_name` only (single text field)
- **Trimodal**: Use `primary_text_name` AND `secondary_text_name` (dual text fields with independent processing)
- These are mutually exclusive - use one approach or the other

#### Model Architecture
| Parameter | Type | Required | Description | Options |
|-----------|------|----------|-------------|---------|
| `model_class` | `str` | Yes | Model architecture to use | `"multimodal_bert"`, `"multimodal_cnn"`, `"bert"`, `"lstm"`, `"multimodal_moe"`, `"multimodal_gate_fusion"`, `"multimodal_cross_attn"` |
| `tokenizer` | `str` | Yes | HuggingFace tokenizer identifier | `"bert-base-multilingual-cased"`, `"bert-base-uncased"` |
| `num_classes` | `int` | Yes | Number of classification classes | `2` (binary), `3+` (multiclass) |
| `is_binary` | `bool` | Yes | Whether task is binary classification | `true`, `false` |
| `multiclass_categories` | `List[Union[int, str]]` | Conditional | Class labels for multiclass (required if is_binary=false) | `[0, 1, 2]`, `["positive", "negative", "neutral"]` |
| `hidden_common_dim` | `int` | No | Hidden dimension for fusion layers | `100` |
| `input_tab_dim` | `int` | No | Tabular input dimension (auto-computed from tab_field_list) | `11` |

#### Text Processing
| Parameter | Type | Required | Description | Range/Default |
|-----------|------|----------|-------------|---------------|
| `max_sen_len` | `int` | Yes | Maximum token length per text chunk | Default: `512` |
| `chunk_trancate` | `bool` | No | Whether to truncate long dialogues | Default: `false` |
| `max_total_chunks` | `int` | No | Maximum chunks per dialogue | Default: `5` |
| `fixed_tokenizer_length` | `bool` | No | Use fixed tokenization length | Default: `true` |
| `text_input_ids_key` | `str` | No | Key name for input IDs in batch | Default: `"input_ids"` |
| `text_attention_mask_key` | `str` | No | Key name for attention mask | Default: `"attention_mask"` |

#### Text Processing Pipeline Configuration (Optional - Advanced)
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `text_processing_steps` | `List[str]` | No | Processing steps for bimodal text field | `["dialogue_splitter", "html_normalizer", "emoji_remover", "text_normalizer", "dialogue_chunker", "tokenizer"]` |
| `primary_text_processing_steps` | `List[str]` | No | Processing steps for primary text (trimodal) | `["dialogue_splitter", "html_normalizer", "emoji_remover", "text_normalizer", "dialogue_chunker", "tokenizer"]` |
| `secondary_text_processing_steps` | `List[str]` | No | Processing steps for secondary text (trimodal) | `["dialogue_splitter", "text_normalizer", "dialogue_chunker", "tokenizer"]` |

**Available Processing Steps**:
- `dialogue_splitter`: Split multi-turn dialogues into individual messages
- `html_normalizer`: Clean HTML tags and entities
- `emoji_remover`: Remove emoji characters
- `text_normalizer`: Normalize whitespace and special characters
- `dialogue_chunker`: Chunk text by token limits
- `tokenizer`: BERT tokenization (always last step)

**Default Pipelines**:
- **Bimodal** (text_name): Full cleaning pipeline with all steps
- **Trimodal Primary** (e.g., chat): Full cleaning pipeline (HTML, emoji, normalization)
- **Trimodal Secondary** (e.g., events): Minimal pipeline (no HTML/emoji removal)

#### Preprocessing Artifact Configuration (Optional - Advanced)
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `smooth_factor` | `float` | No | Risk table smoothing factor for categorical encoding | Default: `0.0` |
| `count_threshold` | `int` | No | Minimum count threshold for risk table mapping | Default: `0` |
| `imputation_dict` | `Dict[str, float]` | No | Pre-computed imputation values (loaded from artifacts) | Auto-computed or loaded |
| `risk_tables` | `Dict[str, Dict]` | No | Pre-computed risk tables (loaded from artifacts) | Auto-computed or loaded |

**Note**: These are automatically populated during training or loaded from pre-computed artifacts

#### Training Hyperparameters
| Parameter | Type | Required | Description | Range/Default |
|-----------|------|----------|-------------|---------------|
| `batch_size` | `int` | Yes | Training batch size | Typical: `16-64` |
| `max_epochs` | `int` | Yes | Maximum training epochs | Typical: `5-50` |
| `lr` | `float` | Yes | Learning rate | Typical: `1e-5 to 0.1` |
| `optimizer` | `str` | Yes | Optimizer type | `"SGD"`, `"Adam"`, `"AdamW"` |
| `momentum` | `float` | No | SGD momentum (if optimizer=SGD) | Default: `0.9` |
| `weight_decay` | `float` | No | L2 regularization weight | Default: `0` |
| `adam_epsilon` | `float` | No | Adam epsilon (if optimizer=Adam/AdamW) | Default: `1e-08` |
| `class_weights` | `List[float]` | No | Class weights for imbalanced data | Example: `[1.0, 10.0]` |
| `dropout_keep` | `float` | No | Dropout keep probability | Range: `0.0-1.0`, Default: `0.5` |
| `gradient_clip_val` | `float` | No | Gradient clipping value | Default: `1.0` |
| `lr_decay` | `float` | No | Learning rate decay rate | Default: `0.05` |

#### CNN-Specific Parameters
| Parameter | Type | Required | Description | Example |
|-----------|------|----------|-------------|---------|
| `kernel_size` | `List[int]` | No | Kernel sizes for CNN | `[3, 5, 7]` |
| `num_layers` | `int` | No | Number of CNN layers | `2` |
| `num_channels` | `List[int]` | No | Channel counts per layer | `[100, 100]` |

#### BERT-Specific Parameters
| Parameter | Type | Required | Description | Default |
|-----------|------|----------|-------------|---------|
| `is_embeddings_trainable` | `bool` | No | Whether to fine-tune BERT embeddings | `true` |
| `reinit_pooler` | `bool` | No | Reinitialize BERT pooler layer | `true` |
| `reinit_layers` | `int` | No | Number of BERT layers to reinitialize | `2` |
| `warmup_steps` | `int` | No | Learning rate warmup steps | `300` |
| `run_scheduler` | `bool` | No | Use learning rate scheduler | `true` |

#### Training Control
| Parameter | Type | Required | Description | Default |
|-----------|------|----------|-------------|---------|
| `early_stop_metric` | `str` | No | Metric for early stopping | `"val/f1_score"` |
| `early_stop_patience` | `int` | No | Early stopping patience (epochs) | `3` |
| `val_check_interval` | `float` | No | Validation check interval (fraction of epoch or batches) | `0.25` |
| `metric_choices` | `List[str]` | No | Metrics to compute | `["auroc", "f1_score"]` |
| `load_ckpt` | `bool` | No | Load best checkpoint for final evaluation | `false` |

#### Advanced Features
| Parameter | Type | Required | Description | Default |
|-----------|------|----------|-------------|---------|
| `fp16` | `bool` | No | Use mixed precision training | `false` |

## Input Data Structure

### Expected Input Format

```
/opt/ml/input/data/
├── train/
│   ├── training_data.csv (or .tsv, .parquet)
│   └── _SUCCESS (optional marker)
├── val/
│   ├── validation_data.csv
│   └── _SUCCESS
├── test/
│   ├── test_data.csv
│   └── _SUCCESS
└── model_artifacts_input/ (optional)
    ├── impute_dict.pkl
    ├── risk_table_map.pkl
    └── selected_features.json

/opt/ml/code/hyperparams/
└── hyperparameters.json
```

### Required Columns in Data Files

**Essential Columns**:
- ID column (name specified by `id_name`): Unique record identifier
- Text column (name specified by `text_name`): Text/dialogue content for processing
- Label column (name specified by `label_name`): Classification target

**Feature Columns**:
- Tabular features (specified in `tab_field_list`): Numerical features for multimodal fusion
- Categorical features (specified in `cat_field_list`): Categorical features requiring encoding

### Supported Data Formats

1. **CSV Files**: Comma-separated values with configurable header
2. **TSV Files**: Tab-separated values
3. **Parquet Files**: Columnar format for efficient storage and loading

### Text Data Requirements

**Dialogue Format**:
- Supports multi-turn dialogues with automatic splitting
- HTML content automatically normalized
- Emojis optionally removed
- Text normalization applied before tokenization

**Chunking Strategy**:
- Long dialogues chunked based on token limits
- Configurable maximum chunks per dialogue
- Optional truncation for very long texts

### Label Requirements

**Binary Classification** (`is_binary=true`):
- Labels must be binary (0/1 or two distinct values)
- `num_classes` must equal 2
- `multiclass_categories` optional (must have 2 items if specified)

**Multiclass Classification** (`is_binary=false`):
- Labels can be integers or strings
- `multiclass_categories` required: explicit list of all class labels
- `num_classes` must match length of `multiclass_categories`
- All values must be unique

## Output Data Structure

### Model Output Directory Structure

```
/opt/ml/model/
├── model.pth                # Trained PyTorch model (state dict)
├── model_artifacts.pth      # Model artifacts bundle
└── model.onnx              # ONNX exported model
```

**model.pth Contents**:
- PyTorch state dictionary
- Model architecture parameters
- Learned weights and biases

**model_artifacts.pth Contents**:
```python
{
    "config": dict,                    # Complete hyperparameter configuration
    "embedding_mat": torch.Tensor,     # Pre-trained embeddings from tokenizer
    "vocab": dict,                     # Tokenizer vocabulary
    "model_class": str,                # Model architecture identifier
    "categorical_processor_mappings": dict,  # Feature encoding mappings
    "label_to_id": dict,              # Label to ID mapping (multiclass only)
    "id_to_label": list               # ID to label mapping (multiclass only)
}
```

**model.onnx**:
- ONNX format for cross-platform deployment
- Compatible with ONNX Runtime
- Preserves model architecture and weights

### Evaluation Output Directory Structure

```
/opt/ml/output/data/
├── predict_results.pth                    # Prediction results
├── tensorboard_eval/                      # TensorBoard logs
│   ├── events.out.tfevents.*
│   ├── roc_curve.png
│   └── pr_curve.png
└── failure (only on error)                # Error details
```

**predict_results.pth Contents**:
```python
{
    "y_true": torch.Tensor,      # True labels
    "y_pred": torch.Tensor       # Predicted probabilities
}
```

**TensorBoard Logs**:
- ROC curves (test and validation)
- Precision-Recall curves
- Metric values (AUROC, F1, Average Precision)
- Training logs and visualizations

### Checkpoint Directory Structure

```
/opt/ml/checkpoints/
├── epoch=N-step=M.ckpt         # Training checkpoints
├── last.ckpt                    # Last checkpoint
└── best.ckpt                    # Best checkpoint (symlink)
```

**Checkpoint Contents**:
- Model state
- Optimizer state
- Training epoch and step
- Metric history

## Key Functions and Tasks

### Data Loading Component

#### `find_first_data_file(data_dir, extensions)`
**Purpose**: Automatically detect the first data file in a directory with supported extensions

**Algorithm**:
```python
1. List all files in data_dir, sorted alphabetically
2. For each file:
   a. Clean filename (strip whitespace, lowercase)
   b. Check if file ends with any extension (.tsv, .csv, .parquet)
   c. If match found, return filename
3. If no match found, raise FileNotFoundError
```

**Parameters**:
- `data_dir` (str): Directory path to search
- `extensions` (List[str]): File extensions to match (default: [".tsv", ".csv", ".parquet"])

**Returns**: `str` - First matching filename

**Example**:
```python
train_filename = find_first_data_file("/opt/ml/input/data/train")
# Returns: "training_data.csv" if exists
```

#### `load_data_module(file_dir, filename, config)`
**Purpose**: Load BSM dataset from file with missing value handling

**Algorithm**:
```python
1. Initialize BSMDataset with config and file path
2. Load data from file (CSV/TSV/Parquet auto-detected)
3. Fill missing values:
   a. Categorical columns: fill with "Unknown"
   b. Numerical columns: fill with appropriate defaults
4. Return initialized dataset
```

**Parameters**:
- `file_dir` (str): Directory containing the data file
- `filename` (str): Name of the data file
- `config` (Config): Configuration object with data specifications

**Returns**: `BSMDataset` - Loaded and preprocessed dataset

### Hyperparameter Management Component

#### `safe_cast(val)`
**Purpose**: Intelligently convert string hyperparameter values to appropriate Python types

**Algorithm**:
```python
1. If value is string:
   a. Strip whitespace
   b. Check for boolean strings ("true", "false") -> bool
   c. Check for JSON structures ([], {}) -> parse with json.loads
   d. Try ast.literal_eval for Python literals
   e. If all fail, return original string
2. If value is not string, return as-is
```

**Type Conversions**:
- `"true"` → `True`
- `"false"` → `False`
- `"[1, 2, 3]"` → `[1, 2, 3]`
- `"{'key': 'value'}"` → `{'key': 'value'}`
- `"123"` → `123`
- `"3.14"` → `3.14`

#### `load_parse_hyperparameters(hparam_path)`
**Purpose**: Load and parse hyperparameters from JSON file with type conversion

**Algorithm**:
```python
1. Open hyperparameters.json file
2. Parse JSON content
3. For each hyperparameter key-value pair:
   a. Apply converter function from converters dict
   b. Log conversion (key, converted value, type)
   c. Store in hyperparameters dict
4. Return complete hyperparameters dictionary
```

**Parameters**:
- `hparam_path` (str): Path to hyperparameters.json file

**Returns**: `Dict` - Parsed and type-converted hyperparameters

**Example Output**:
```python
{
    "batch_size": 32,                              # Converted to int
    "lr": 0.02,                                    # Converted to float
    "is_binary": True,                             # Converted to bool
    "kernel_size": [3, 5, 7],                      # Converted to list
    "tokenizer": "bert-base-multilingual-cased"    # Remains string
}
```

### Text Preprocessing Pipeline Component

#### `data_preprocess_pipeline(config)`
**Purpose**: Construct complete text preprocessing pipeline with tokenization

**Algorithm**:
```python
1. Load tokenizer from HuggingFace (e.g., BERT tokenizer)
2. Build dialogue processing pipeline:
   a. DialogueSplitterProcessor: Split multi-turn dialogues
   b. HTMLNormalizerProcessor: Clean HTML tags and entities
   c. EmojiRemoverProcessor: Remove emoji characters
   d. TextNormalizationProcessor: Normalize whitespace and special chars
   e. DialogueChunkerProcessor: Chunk long texts based on token limits
   f. BertTokenizeProcessor: Convert text to BERT input format
3. Return tokenizer and processing pipeline
```

**Pipeline Flow**:
```
Raw Dialogue Text
    ↓ (split multi-turn)
Separated Messages
    ↓ (clean HTML)
Normalized Text
    ↓ (remove emojis)
Clean Text
    ↓ (normalize)
Standardized Text
    ↓ (chunk by tokens)
Text Chunks
    ↓ (tokenize)
BERT Input Format {input_ids, attention_mask}
```

**Parameters**:
- `config` (Config): Configuration with tokenizer name and text processing settings

**Returns**: `Tuple[AutoTokenizer, Dict[str, Processor]]` - Tokenizer and processing pipelines

**Output Format**:
```python
{
    "input_ids": [[101, 2023, 2003, ..., 102], ...],      # BERT token IDs
    "attention_mask": [[1, 1, 1, ..., 1], ...]            # Attention mask
}
```

### Categorical Feature Encoding Component

#### `build_categorical_label_pipelines(config, datasets)`
**Purpose**: Build categorical label processors for all categorical features across all datasets

**Algorithm**:
```python
1. Extract categorical features to encode from config
2. For each categorical feature:
   a. Collect all unique values across train/val/test datasets
   b. Sort unique values for consistent ordering
   c. Create CategoricalLabelProcessor with unique values
   d. Store processor in field_to_processor dict
3. Return mapping of field names to processors
```

**Parameters**:
- `config` (Config): Configuration with categorical_features_to_encode list
- `datasets` (List[BSMDataset]): List of train/val/test datasets

**Returns**: `Dict[str, CategoricalLabelProcessor]` - Mapping of field names to label processors

**Encoding Strategy**:
- Consistent encoding across all splits (train/val/test)
- Unknown values at inference time map to special token
- Preserves original categorical semantics

**Example**:
```python
# Input categorical feature values across datasets
marketplace: ["US", "UK", "DE", "US", "FR", "UK"]

# Processor output
{
    "marketplace": CategoricalLabelProcessor(
        categories=["DE", "FR", "UK", "US"],  # Sorted
        category_to_label={"DE": 0, "FR": 1, "UK": 2, "US": 3}
    )
}
```

### Model Selection Component

#### `model_select(model_class, config, vocab_size, embedding_mat)`
**Purpose**: Instantiate appropriate model architecture based on model_class parameter

**Supported Architectures**:

1. **multimodal_bert**: BERT + tabular fusion with concatenation
   - Text encoder: BERT transformer
   - Tabular encoder: Feed-forward network
   - Fusion: Concatenate text and tabular representations
   - Use case: Balanced text and tabular importance

2. **multimodal_cnn**: CNN + tabular fusion
   - Text encoder: Multi-kernel CNN
   - Tabular encoder: Feed-forward network
   - Fusion: Concatenate CNN features and tabular
   - Use case: When text patterns more important than semantics

3. **multimodal_moe**: Mixture of Experts multimodal
   - Text encoder: BERT
   - Tabular encoder: Feed-forward network
   - Fusion: Gated mixture of expert networks
   - Use case: Complex interactions between modalities

4. **multimodal_gate_fusion**: Gated fusion multimodal
   - Text encoder: BERT
   - Tabular encoder: Feed-forward network
   - Fusion: Learned gating mechanism
   - Use case: Adaptive modality weighting

5. **multimodal_cross_attn**: Cross-attention multimodal
   - Text encoder: BERT
   - Tabular encoder: Feed-forward network
   - Fusion: Cross-attention between modalities
   - Use case: Fine-grained modality interactions

6. **bert**: Text-only BERT classification
   - Single BERT encoder with classification head
   - Use case: Text-only tasks, baseline comparison

7. **lstm**: Text-only LSTM classification
   - LSTM encoder with classification head
   - Use case: Sequential text modeling, lightweight alternative

**Parameters**:
- `model_class` (str): Model architecture identifier
- `config` (Config): Full configuration object
- `vocab_size` (int): Tokenizer vocabulary size
- `embedding_mat` (torch.Tensor): Pre-trained embeddings

**Returns**: `nn.Module` - Instantiated PyTorch model

**Selection Logic**:
```python
if model_class == "multimodal_cnn":
    return MultimodalCNN(config, vocab_size, embedding_mat)
elif model_class == "bert":
    return TextBertClassification(config)
elif model_class == "lstm":
    return TextLSTM(config, vocab_size, embedding_mat)
elif model_class == "multimodal_bert":
    return MultimodalBert(config)
elif model_class == "multimodal_moe":
    return MultimodalBertMoE(config)
elif model_class == "multimodal_gate_fusion":
    return MultimodalBertGateFusion(config)
elif model_class == "multimodal_cross_attn":
    return MultimodalBertCrossAttn(config)
else:
    return TextBertClassification(config)  # Default fallback
```

### Training Component

#### `model_train(model, config, train_dataloader, val_dataloader, device, model_log_path, early_stop_metric)`
**Purpose**: Train model using PyTorch Lightning with comprehensive training features

**Training Features**:
- Automatic GPU utilization
- Distributed training support (FSDP, DDP)
- Early stopping based on validation metrics
- Model checkpointing (best and last)
- TensorBoard logging
- Gradient clipping
- Learning rate scheduling
- Mixed precision training (optional)

**Algorithm**:
```python
1. Initialize Lightning Trainer with:
   a. Accelerator (GPU/CPU auto-detected)
   b. Max epochs from config
   c. Validation check interval
   d. Gradient clipping
   e. Precision (fp16 if enabled)
2. Setup callbacks:
   a. EarlyStopping: monitor early_stop_metric with patience
   b. ModelCheckpoint: save best model based on metric
   c. TensorBoard logger: log metrics and hyperparameters
3. Call trainer.fit(model, train_dataloader, val_dataloader)
4. Return trained model and trainer object
```

**Parameters**:
- `model` (nn.Module): Model to train
- `config` (dict): Configuration dictionary
- `train_dataloader` (DataLoader): Training data loader
- `val_dataloader` (DataLoader): Validation data loader
- `device` (str): Device specification ("auto", "cuda", "cpu")
- `model_log_path` (str): Path for checkpoints and logs
- `early_stop_metric` (str): Metric for early stopping

**Returns**: `pl.Trainer` - Trained Lightning trainer object

#### `model_inference(model, dataloader, accelerator, device, model_log_path)`
**Purpose**: Run inference on data loader and collect predictions

**Algorithm**:
```python
1. Initialize Lightning Trainer in prediction mode
2. Call trainer.predict(model, dataloader)
3. Extract predictions and true labels from batch outputs
4. Convert to tensors and concatenate across batches
5. Return predicted probabilities and true labels
```

**Parameters**:
- `model` (nn.Module): Trained model
- `dataloader` (DataLoader): Data loader for inference
- `accelerator` (str): Accelerator type
- `device` (str): Device specification
- `model_log_path` (str): Path for logs

**Returns**: `Tuple[torch.Tensor, torch.Tensor]` - (predicted probabilities, true labels)

### Model Export Component

#### `export_model_to_onnx(model, trainer, val_dataloader, onnx_path)`
**Purpose**: Export trained PyTorch model to ONNX format for deployment

**Algorithm**:
```python
1. Sample batch from validation dataloader
2. Move batch to CPU for export
3. If model wrapped in FSDP:
   a. Unwrap to get original model
   b. Log unwrapping operation
4. Move model to CPU and set to eval mode
5. Call model.export_to_onnx(onnx_path, sample_batch)
6. Log successful export or raise error
```

**FSDP Handling**:
- Detects FSDP-wrapped models
- Unwraps to access original model
- Ensures clean export without distributed training artifacts

**Parameters**:
- `model` (torch.nn.Module): Trained model (may be FSDP-wrapped)
- `trainer` (pl.Trainer): Lightning trainer (for strategy checking)
- `val_dataloader` (DataLoader): Data loader for sampling
- `onnx_path` (Union[str, Path]): Output path for ONNX model

**Returns**: None (saves model to onnx_path)

**Example**:
```python
export_model_to_onnx(
    model=trained_model,
    trainer=trainer,
    val_dataloader=val_loader,
    onnx_path="/opt/ml/model/model.onnx"
)
```

### Evaluation and Metrics Component

#### `compute_metrics(y_pred, y_true, output_metrics, task, num_classes, stage)`
**Purpose**: Compute comprehensive classification metrics for model evaluation

**Supported Metrics**:
- `"auroc"`: Area Under ROC Curve
- `"average_precision"`: Average Precision (PR-AUC)
- `"f1_score"`: F1 Score
- `"precision"`: Precision
- `"recall"`: Recall
- `"accuracy"`: Accuracy

**Algorithm**:
```python
1. Convert predictions and labels to appropriate format
2. For each requested metric:
   a. Compute metric value using appropriate library (torchmetrics/sklearn)
   b. Handle binary vs multiclass differences
   c. Store result with stage prefix (e.g., "val/auroc")
3. Return dictionary of metric name to value
```

**Parameters**:
- `y_pred` (torch.Tensor): Predicted probabilities
- `y_true` (torch.Tensor): True labels
- `output_metrics` (List[str]): Metrics to compute
- `task` (str): "binary" or "multiclass"
- `num_classes` (int): Number of classes
- `stage` (str): "train", "val", or "test"

**Returns**: `Dict[str, float]` - Metric name to value mapping

**Example Output**:
```python
{
    "val/auroc": 0.8542,
    "val/f1_score": 0.7823,
    "val/average_precision": 0.8012
}
```

#### `roc_metric_plot(y_pred, y_true, y_val_pred, y_val_true, path, task, num_classes, writer, global_step)`
**Purpose**: Generate and save ROC curve visualization

**Algorithm**:
```python
1. Compute ROC curve points for test set
2. Compute ROC curve points for validation set
3. Create matplotlib figure with both curves
4. Calculate AUC for both sets
5. Save figure to file
6. Log to TensorBoard
```

**Parameters**:
- `y_pred` (torch.Tensor): Test predictions
- `y_true` (torch.Tensor): Test labels
- `y_val_pred` (torch.Tensor): Validation predictions
- `y_val_true` (torch.Tensor): Validation labels
- `path` (str): Output directory path
- `task` (str): "binary" or "multiclass"
- `num_classes` (int): Number of classes
- `writer` (SummaryWriter): TensorBoard writer
- `global_step` (int): Training step for logging

**Returns**: None (saves plot and logs to TensorBoard)

#### `pr_metric_plot(y_pred, y_true, y_val_pred, y_val_true, path, task, num_classes, writer, global_step)`
**Purpose**: Generate and save Precision-Recall curve visualization

**Algorithm**:
```python
1. Compute PR curve points for test set
2. Compute PR curve points for validation set
3. Create matplotlib figure with both curves
4. Calculate Average Precision for both sets
5. Save figure to file
6. Log to TensorBoard
```

**Parameters**: Same as `roc_metric_plot`

**Returns**: None (saves plot and logs to TensorBoard)

### Main Orchestration Component

#### `load_and_preprocess_data(config)`
**Purpose**: Orchestrate complete data loading and preprocessing workflow

**Algorithm**:
```python
1. Determine filenames for train/val/test:
   a. Use explicit filenames from config if provided
   b. Otherwise auto-detect first matching file in each directory
2. Load raw BSM datasets for train/val/test
3. Build text preprocessing pipeline:
   a. Create tokenizer
   b. Construct dialogue processing pipeline
   c. Add to each dataset
4. Build categorical feature encoders:
   a. Collect unique values across all datasets
   b. Create label processors
   c. Add to each dataset
5. If multiclass classification:
   a. Create MultiClassLabelProcessor with categories
   b. Add to label column in each dataset
   c. Store label mappings in config
6. Store categorical processor mappings in config
7. Return datasets, tokenizer, and updated config
```

**Parameters**:
- `config` (Config): Configuration object

**Returns**: `Tuple[List[BSMDataset], AutoTokenizer, Config]` - Datasets, tokenizer, updated config

#### `build_model_and_optimizer(config, tokenizer, datasets)`
**Purpose**: Build model architecture and data loaders

**Algorithm**:
```python
1. Create collate function with configurable key names
2. Build DataLoaders for train/val/test:
   a. Set batch size from config
   b. Shuffle training data
   c. Apply collate function
3. Extract pre-trained embeddings from tokenizer:
   a. Load AutoModel for tokenizer
   b. Extract word embedding matrix
   c. Store embedding size in config
4. Select and instantiate model architecture:
   a. Use model_select with model_class
   b. Pass config, vocab_size, embeddings
5. Return model and data loaders
```

**Parameters**:
- `config` (Config): Configuration object
- `tokenizer` (AutoTokenizer): Initialized tokenizer
- `datasets` (List[BSMDataset]): Loaded datasets

**Returns**: `Tuple[nn.Module, DataLoader, DataLoader, DataLoader, torch.Tensor]` - Model, train/val/test loaders, embeddings

#### `evaluate_and_log_results(model, val_dataloader, test_dataloader, config, trainer)`
**Purpose**: Comprehensive model evaluation with metrics and visualizations

**Algorithm**:
```python
1. Run inference on validation set:
   a. Get predicted probabilities
   b. Get true labels
2. Run inference on test set:
   a. Get predicted probabilities
   b. Get true labels
3. Compute metrics for both sets:
   a. AUROC, F1, Average Precision
   b. Log values to console
4. Generate visualizations:
   a. ROC curves (test and validation)
   b. Precision-Recall curves
   c. Save to output directory
   d. Log to TensorBoard
5. Save prediction results to file
```

**Parameters**:
- `model` (nn.Module): Trained model
- `val_dataloader` (DataLoader): Validation data loader
- `test_dataloader` (DataLoader): Test data loader
- `config` (Config): Configuration object
- `trainer` (pl.Trainer): Lightning trainer

**Returns**: None (saves results and logs)

#### `main(input_paths, output_paths, environ_vars, job_args)`
**Purpose**: Main entry point that orchestrates complete training workflow

**Workflow**:
```
1. Load & Validate Hyperparameters
   ↓
2. Setup Training Environment
   ↓
3. Load & Preprocess Data
   ↓
4. Build Model & Data Loaders
   ↓
5. Train Model (Lightning)
   ↓
6. Load Best Checkpoint (if requested)
   ↓
7. Save Model & Artifacts
   ↓
8. Export to ONNX
   ↓
9. Evaluate & Generate Metrics
   ↓
10. Save Visualizations & Results
```

**Parameters**:
- `input_paths` (Dict[str, str]): Input path mapping
- `output_paths` (Dict[str, str]): Output path mapping
- `environ_vars` (Dict[str, str]): Environment variables
- `job_args` (argparse.Namespace): Command line arguments

**Returns**: None (exits with code 0 on success, 1 on failure)

## Algorithms and Data Structures

### Distributed Training Patterns

PyTorch distributed training introduces complexities around process synchronization, data distribution, and resource management. This section documents key patterns and best practices for robust multi-GPU training.

#### 1. Barrier Synchronization Pattern

**Problem**: In multi-GPU distributed training, non-main ranks may exit prematurely before the main process completes evaluation and file writing, causing:
- Incomplete prediction files
- Missing evaluation metrics
- Array length mismatches
- Training reported as "complete" when evaluation hasn't finished

**Solution Strategy**:
1. Add barrier after training before evaluation starts
2. Only main process runs evaluation and saves files
3. Add final barrier before script exit to ensure main process completes

**Algorithm**:
```python
# After training completes
if torch.distributed.is_initialized():
    torch.distributed.barrier()
    log_once(logger, "All ranks synchronized after training - proceeding to evaluation")

# Only main process runs evaluation
if is_main_process():
    log_once(logger, "Main process starting evaluation and prediction saving...")
    evaluate_and_log_results(
        model, val_dataloader, test_dataloader, config, trainer,
        val_dataset, test_dataset, paths
    )
    log_once(logger, "Evaluation and prediction saving complete")
else:
    log_once(logger, f"Rank {get_rank()} skipping evaluation (main process only)")

# CRITICAL: Final barrier ensures main process completes before any rank exits
if torch.distributed.is_initialized():
    torch.distributed.barrier()
    log_once(logger, "All ranks synchronized after evaluation - ready to exit")
```

**Complexity**: O(1) - barrier synchronization

**Key Features**:
- Prevents race conditions in distributed training
- Ensures file writes complete before process exit
- No performance impact (barriers at end of training)
- Compatible with both DDP and FSDP strategies

**When This Matters**:
- Multi-GPU training (2+ GPUs)
- Distributed training with DDP or FSDP
- Long-running evaluation steps
- Large prediction file writes

#### 2. Rank-Aware Execution Pattern

**Problem**: In distributed training, certain operations should only execute once (e.g., saving models, logging metrics, creating directories) to avoid:
- Duplicate file writes and race conditions
- Excessive logging cluttering output
- Wasted compute on redundant operations

**Solution Strategy**:
1. Define helper functions to check rank status
2. Guard single-execution operations with rank checks
3. Use log_once for rank-aware logging

**Algorithm**:
```python
def get_rank():
    """Get current process rank in distributed training."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return 0

def is_main_process():
    """Check if current process is the main process (rank 0)."""
    return get_rank() == 0

def log_once(logger, message, level=logging.INFO):
    """Log message only from main process."""
    if is_main_process():
        logger.log(level, message)

# Usage examples
if is_main_process():
    # Only main process saves model
    save_model(model_path, model)
    
    # Only main process creates directories
    os.makedirs(output_dir, exist_ok=True)
    
    # Only main process logs metrics
    log_once(logger, f"Training complete. Final loss: {loss}")
```

**Complexity**: O(1) - simple rank check

**Key Features**:
- Prevents duplicate operations across ranks
- Reduces I/O contention
- Cleaner log output
- Saves compute resources

**Common Use Cases**:
- Model checkpoint saving
- Final metrics computation and logging
- Directory creation
- File writing operations
- Visualization generation

#### 3. Data Distribution Pattern

**Problem**: In distributed training, data must be properly distributed across GPUs to ensure:
- Each rank processes unique data samples (no duplication)
- All data is processed exactly once per epoch
- Balanced load across all GPUs

**Solution Strategy**:
1. Use DistributedSampler for automatic data sharding
2. Ensure consistent batch sizes across ranks
3. Handle edge cases (dataset size not divisible by world size)

**Algorithm**:
```python
from torch.utils.data.distributed import DistributedSampler

# During DataLoader creation
if torch.distributed.is_initialized():
    sampler = DistributedSampler(
        dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=True,  # For training data
        drop_last=False  # Keep all samples
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use distributed sampler
        shuffle=False,  # Shuffle handled by sampler
        num_workers=4,
        pin_memory=True
    )
else:
    # Single GPU/CPU training
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

# Set epoch for sampler (important for proper shuffling)
if hasattr(dataloader.sampler, 'set_epoch'):
    dataloader.sampler.set_epoch(epoch)
```

**Complexity**: O(1) per sample - efficient sharding

**Key Features**:
- Automatic data partitioning across ranks
- Consistent shuffling with different random seed per epoch
- Handles uneven data splits gracefully
- No manual data distribution needed

**Best Practices**:
- Always call `sampler.set_epoch(epoch)` before each epoch
- Use `drop_last=False` to avoid losing tail samples
- Ensure batch size is reasonable for each rank
- Consider gradient accumulation for effective larger batch sizes

#### 4. Gradient Synchronization Pattern (DDP)

**Problem**: In DistributedDataParallel (DDP), gradients must be synchronized across ranks to ensure consistent model updates.

**Solution Strategy**:
1. DDP automatically handles gradient synchronization during backward pass
2. Use gradient accumulation for larger effective batch sizes
3. Proper gradient clipping before optimizer step

**Algorithm**:
```python
# DDP wrapping (handled by Lightning, but shown for understanding)
if torch.distributed.is_initialized():
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        find_unused_parameters=False  # Set True if dynamic computation graph
    )

# Training loop with gradient accumulation
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(dataloader):
    # Forward pass
    outputs = model(batch)
    loss = criterion(outputs, targets)
    
    # Scale loss for gradient accumulation
    loss = loss / accumulation_steps
    
    # Backward pass (gradients accumulated locally)
    loss.backward()
    
    # Update weights every accumulation_steps
    if (i + 1) % accumulation_steps == 0:
        # Clip gradients (happens after DDP gradient sync)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
```

**Complexity**: 
- Gradient synchronization: O(model_size) per backward pass
- All-reduce operation across GPUs

**Key Features**:
- Automatic gradient averaging across ranks
- Synchronization happens during backward pass
- Supports gradient accumulation transparently
- Efficient ring all-reduce communication

**When to Use**:
- Standard distributed training (< 1B parameters)
- When model fits in single GPU memory
- Need deterministic gradient synchronization

#### 5. FSDP Memory Optimization Pattern

**Problem**: Fully Sharded Data Parallel (FSDP) enables training very large models by sharding model parameters, gradients, and optimizer states across GPUs.

**Solution Strategy**:
1. Use FSDP for models that don't fit in single GPU
2. Configure sharding strategy based on model size
3. Handle state dict saving/loading correctly

**Algorithm**:
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy

# FSDP configuration (handled by Lightning, shown for understanding)
fsdp_config = {
    "sharding_strategy": ShardingStrategy.FULL_SHARD,  # Shard params, grads, optimizer
    "cpu_offload": False,  # Set True for very large models
    "mixed_precision": True,  # Enable automatic mixed precision
}

# Wrap model with FSDP
if torch.distributed.is_initialized():
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=local_rank,
        mixed_precision=mixed_precision_policy,
    )

# Saving checkpoint (only main rank)
if is_main_process():
    # For FSDP, need to gather full state dict
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        state_dict = model.state_dict()
        torch.save(state_dict, checkpoint_path)
```

**Complexity**: 
- Memory: O(model_size / world_size) per rank
- Communication: O(model_size) for state dict gathering

**Key Features**:
- Shard model parameters across GPUs
- Reduced per-GPU memory footprint
- Supports models up to 100B+ parameters
- Automatic sharding and gathering

**When to Use**:
- Large models (> 1B parameters)
- Memory-constrained training
- Need to maximize model size on available GPUs

**Sharding Strategies**:
- `FULL_SHARD`: Maximum memory savings (shard everything)
- `SHARD_GRAD_OP`: Shard gradients and optimizer states only
- `NO_SHARD`: DDP equivalent (no sharding)

#### 6. Checkpoint Saving Pattern

**Problem**: In distributed training, checkpoints should be saved only once to avoid:
- Duplicate checkpoint files from each rank
- Race conditions during file writes
- Wasted I/O bandwidth

**Solution Strategy**:
1. Only main rank saves checkpoints
2. Add barrier before save to ensure all ranks ready
3. Add barrier after save to ensure file complete

**Algorithm**:
```python
def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    """Save checkpoint in distributed training."""
    # Barrier before save: ensure all ranks ready
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    # Only main rank saves
    if is_main_process():
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(checkpoint, checkpoint_path)
        log_once(logger, f"Checkpoint saved: {checkpoint_path}")
    
    # Barrier after save: ensure file complete before proceeding
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

def load_checkpoint(model, optimizer, checkpoint_path):
    """Load checkpoint in distributed training."""
    # Load on all ranks (model parameters synced by DDP/FSDP)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Barrier to ensure all ranks loaded
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    return checkpoint['epoch']
```

**Complexity**: O(model_size) for I/O on main rank

**Key Features**:
- Single checkpoint file (not per-rank files)
- Synchronized checkpoint saving
- Safe for concurrent access

**Best Practices**:
- Save to fast storage (local SSD > network storage)
- Use atomic writes (write to temp file, then rename)
- Include optimizer state for resuming training
- Keep multiple checkpoints for safety

#### 7. Error Handling Pattern

**Problem**: In distributed training, an error in one rank can hang other ranks waiting at barriers or cause cascading failures.

**Solution Strategy**:
1. Wrap training loop in try-except
2. Propagate errors to all ranks
3. Clean up distributed processes properly

**Algorithm**:
```python
def distributed_training_with_error_handling():
    """Training with proper error handling for distributed setup."""
    try:
        # Setup distributed training
        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        
        # Main training loop
        for epoch in range(max_epochs):
            train_one_epoch(model, dataloader, optimizer)
            
            # Barrier after each epoch
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
        
        # Final barrier before cleanup
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        return True
        
    except Exception as e:
        # Log error with rank information
        if torch.distributed.is_initialized():
            logger.error(f"Rank {torch.distributed.get_rank()} encountered error: {str(e)}")
        else:
            logger.error(f"Training error: {str(e)}")
        
        # Print full traceback
        import traceback
        logger.error(traceback.format_exc())
        
        # Clean up distributed resources
        if torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except:
                pass
        
        # Re-raise or exit
        raise
        
    finally:
        # Cleanup code runs regardless of success/failure
        if torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except:
                pass
```

**Complexity**: O(1) - error handling overhead

**Key Features**:
- Graceful error handling across ranks
- Proper resource cleanup
- Detailed error logging with rank info
- Prevents hanging processes

**Common Errors**:
- NCCL communication failures: Check network, GPU connectivity
- CUDA out of memory: Reduce batch size or model size
- Timeout errors: Increase timeout in init_process_group
- Deadlock at barrier: Check for rank-specific code paths

#### 8. Logging Pattern

**Problem**: With multiple ranks, logging can be overwhelming with duplicate messages from each rank.

**Solution Strategy**:
1. Implement rank-aware logging function
2. Log only from main rank by default
3. Include rank info for debugging

**Algorithm**:
```python
import logging

# Setup logger with rank-aware configuration
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Only add handler on main process
if is_main_process():
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False

def log_once(logger, message, level=logging.INFO):
    """Log only from main process."""
    if is_main_process():
        logger.log(level, message)

def log_all_ranks(logger, message, level=logging.INFO):
    """Log from all ranks with rank prefix."""
    rank = get_rank() if torch.distributed.is_initialized() else 0
    logger.log(level, f"[Rank {rank}] {message}")

# Usage examples
log_once(logger, "Training starting...")  # Only main rank logs
log_once(logger, f"Epoch {epoch} completed")

# For debugging, log from all ranks
log_all_ranks(logger, f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
```

**Complexity**: O(1) - simple rank check

**Key Features**:
- Clean log output without duplication
- Optional per-rank logging for debugging
- Consistent logging across distributed code

**Best Practices**:
- Use log_once for high-level progress messages
- Use log_all_ranks for debugging memory/performance issues
- Include rank in error messages
- Log before and after barriers for debugging deadlocks

#### 9. Memory Management Pattern

**Problem**: CUDA memory fragmentation in distributed training can cause out-of-memory errors even when sufficient memory exists.

**Solution Strategy**:
1. Monitor memory usage per rank
2. Empty cache at strategic points
3. Use gradient checkpointing for large models

**Algorithm**:
```python
import torch

def log_memory_stats(logger, stage=""):
    """Log CUDA memory statistics for debugging."""
    if torch.cuda.is_available():
        rank = get_rank() if torch.distributed.is_initialized() else 0
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        
        logger.info(
            f"[Rank {rank}] {stage} - "
            f"Allocated: {allocated:.2f}GB, "
            f"Reserved: {reserved:.2f}GB, "
            f"Peak: {max_allocated:.2f}GB"
        )

# Training loop with memory management
for epoch in range(max_epochs):
    # Log memory at epoch start
    log_memory_stats(logger, f"Epoch {epoch} start")
    
    for batch_idx, batch in enumerate(dataloader):
        # Training step
        loss = training_step(model, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # Periodic memory cleanup
        if batch_idx % 100 == 0:
            torch.cuda.empty_cache()
            
    # Barrier and memory cleanup after epoch
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    
    torch.cuda.empty_cache()
    log_memory_stats(logger, f"Epoch {epoch} end")

# Enable gradient checkpointing for large models
if hasattr(model, 'gradient_checkpointing_enable'):
    model.gradient_checkpointing_enable()
```

**Complexity**: 
- empty_cache(): O(1) - fast cache cleanup
- gradient checkpointing: Trade compute for memory

**Key Features**:
- Monitor memory usage per rank
- Strategic cache cleanup
- Gradient checkpointing for memory savings
- Debug memory leaks and fragmentation

**Best Practices**:
- Call empty_cache() between epochs, not during training
- Use gradient checkpointing for models > 1B parameters
- Monitor peak memory usage to optimize batch size
- Set PYTORCH_CUDA_ALLOC_CONF for better memory management

### Pydantic Configuration Validation

**Problem**: Ensure hyperparameter consistency and validity before training begins

**Solution Strategy**:
1. Define comprehensive Config schema with Pydantic
2. Use field validators for complex constraints
3. Implement model_post_init for cross-field validation

**Algorithm**:
```python
class Config(BaseModel):
    # Field definitions with types and defaults
    num_classes: int
    is_binary: bool
    multiclass_categories: List[Union[int, str]]
    class_weights: List[float]
    
    def model_post_init(self, __context):
        # Validate binary classification constraints
        if self.is_binary and self.num_classes != 2:
            raise ValueError("For binary classification, num_classes must be 2")
        
        # Validate multiclass constraints
        if not self.is_binary:
            if self.num_classes < 2:
                raise ValueError("num_classes must be >= 2 for multiclass")
            if not self.multiclass_categories:
                raise ValueError("multiclass_categories required")
            if len(self.multiclass_categories) != self.num_classes:
                raise ValueError("num_classes must match multiclass_categories length")
            if len(set(self.multiclass_categories)) != len(self.multiclass_categories):
                raise ValueError("multiclass_categories must be unique")
        
        # Validate class weights
        if self.class_weights and len(self.class_weights) != self.num_classes:
            raise ValueError("class_weights must match num_classes")
```

**Complexity**: O(n) where n is number of categories

**Key Features**:
- Early error detection before training
- Clear error messages with actionable feedback
- Type safety and automatic conversion

### Text Preprocessing Pipeline

**Problem**: Convert raw text dialogues into BERT-compatible input format with proper chunking

**Solution Strategy**:
1. Chain multiple processors using >> operator
2. Each processor transforms data and passes to next
3. Final output is BERT input_ids and attention_mask

**Pipeline Architecture**:
```python
DialogueSplitterProcessor()  # Split multi-turn dialogues
    ↓
HTMLNormalizerProcessor()    # Clean HTML
    ↓
EmojiRemoverProcessor()      # Remove emojis
    ↓
TextNormalizationProcessor() # Normalize whitespace
    ↓
DialogueChunkerProcessor()   # Chunk by tokens
    ↓
BertTokenizeProcessor()      # Tokenize to BERT format
```

**Complexity**:
- Time: O(n * m) where n=number of dialogues, m=average dialogue length
- Space: O(n * k) where k=max_total_chunks

**Key Features**:
- Modular design with independent processors
- Configurable chunking strategy
- Handles variable-length inputs

### Categorical Feature Encoding

**Problem**: Consistently encode categorical features across train/val/test splits

**Solution Strategy**:
1. Collect all unique values from all datasets
2. Create sorted, deterministic encoding
3. Apply same encoding to all splits

**Algorithm**:
```python
for field in categorical_features:
    all_values = []
    for dataset in [train, val, test]:
        values = dataset[field].dropna().astype(str).tolist()
        all_values.extend(values)
    
    unique_values = sorted(set(all_values))
    processor = CategoricalLabelProcessor(initial_categories=unique_values)
    
    # Apply to all datasets
    for dataset in [train, val, test]:
        dataset.add_pipeline(field, processor)
```

**Complexity**: O(n * m) where n=number of features, m=total dataset size

**Key Features**:
- No train/test leakage
- Consistent encoding across splits
- Handles unknown categories at inference

### Multiclass Label Processing

**Problem**: Convert arbitrary label values to sequential integer IDs for PyTorch

**Solution Strategy**:
1. Accept user-defined label list in order
2. Create bidirectional mapping (label ↔ ID)
3. Store mappings for inference-time decoding

**Algorithm**:
```python
if not is_binary:
    label_processor = MultiClassLabelProcessor(
        label_list=config.multiclass_categories,
        strict=True  # Reject unknown labels
    )
    
    # Apply to all datasets
    for dataset in [train, val, test]:
        dataset.add_pipeline(config.label_name, label_processor)
    
    # Store mappings in config for deployment
    config.label_to_id = label_processor.label_to_id
    config.id_to_label = label_processor.id_to_label
```

**Mappings Example**:
```python
# Input labels: ["positive", "negative", "neutral"]
label_to_id = {"positive": 0, "negative": 1, "neutral": 2}
id_to_label = ["positive", "negative", "neutral"]
```

**Complexity**: O(n) for n labels

**Key Features**:
- Preserves label semantics
- Bidirectional lookup
- Strict validation prevents errors

## Performance Characteristics

### Training Performance

| Configuration | Batch Size | GPU Memory | Training Speed | Best For |
|---------------|------------|------------|----------------|----------|
| Single GPU | 32 | ~8GB | Baseline | Small datasets (<100K) |
| Single GPU + fp16 | 64 | ~8GB | 1.5-2x faster | Medium datasets (100K-1M) |
| Multi-GPU (FSDP) | 32/GPU | Distributed | ~Linear speedup | Large datasets (>1M) |
| Multi-GPU + fp16 | 64/GPU | Distributed | Best throughput | Production training |

### Model Complexity

| Model Architecture | Parameters | Inference Time | GPU Memory |
|-------------------|------------|----------------|------------|
| bert | ~110M | ~10ms | ~2GB |
| lstm | ~10M | ~5ms | ~1GB |
| multimodal_bert | ~115M | ~12ms | ~2.5GB |
| multimodal_cnn | ~15M | ~8ms | ~1.5GB |
| multimodal_moe | ~150M | ~15ms | ~3GB |
| multimodal_gate_fusion | ~120M | ~13ms | ~2.5GB |
| multimodal_cross_attn | ~130M | ~14ms | ~3GB |

### Data Loading Performance

| Dataset Size | Format | Load Time | Preprocessing Time |
|--------------|--------|-----------|-------------------|
| 10K samples | CSV | ~2s | ~10s |
| 10K samples | Parquet | ~0.5s | ~10s |
| 100K samples | CSV | ~20s | ~100s |
| 100K samples | Parquet | ~3s | ~100s |
| 1M samples | Parquet | ~30s | ~15min |

**Optimization Tips**:
- Use Parquet for faster I/O (4-10x speedup)
- Enable num_workers in DataLoader for parallel loading
- Cache preprocessed data for multiple epochs
- Use fp16 for 1.5-2x training speedup

## Error Handling

### Configuration Validation Errors

**ValidationError: num_classes mismatch**
```python
ValidationError: num_classes=3 does not match len(multiclass_categories)=2
```
**Cause**: Mismatch between num_classes and multiclass_categories length

**Handling**: Pydantic raises ValidationError with clear message before training starts

**Resolution**: Ensure num_classes matches the length of multiclass_categories

**ValidationError: multiclass_categories must be unique**
```python
ValidationError: multiclass_categories must contain unique values
```
**Cause**: Duplicate values in multiclass_categories

**Handling**: Early detection during config validation

**Resolution**: Remove duplicates from multiclass_categories list

### Data Loading Errors

**FileNotFoundError: No data file found**
```python
FileNotFoundError: No supported data file (.tsv, .csv, .parquet) found in /opt/ml/input/data/train
```
**Cause**: Missing data file in expected directory

**Handling**: Raised during file detection with clear message

**Resolution**: Ensure data files exist with supported extensions or specify explicit filename

**KeyError: Missing required column**
```python
KeyError: Column 'text' not found in dataset
```
**Cause**: Required column (id_name, text_name, label_name) missing from data

**Handling**: Pandas raises KeyError during dataset loading

**Resolution**: Verify column names in data match config parameters

### Model Training Errors

**RuntimeError: CUDA out of memory**
```python
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```
**Cause**: Batch size or model too large for available GPU memory

**Handling**: PyTorch raises RuntimeError during forward pass

**Resolution**: Reduce batch_size, enable fp16, or use gradient accumulation

**ValueError: Invalid model_class**
```python
ValueError: Unknown model class: invalid_model
```
**Cause**: Unsupported model_class value

**Handling**: Caught during model selection

**Resolution**: Use one of the 7 supported model classes

### Export Errors

**RuntimeError: ONNX export failed**
```python
RuntimeError: Failed to export model to ONNX
```
**Cause**: Model architecture incompatible with ONNX or missing operators

**Handling**: Exception caught and logged with full traceback

**Resolution**: Check model architecture compatibility, simplify custom layers

## Best Practices

### For Production Deployments

1. **Use Mixed Precision Training**
   ```python
   {
       "fp16": true,
       "gradient_clip_val": 1.0
   }
   ```
   - Reduces memory usage by 40-50%
   - Speeds up training by 1.5-2x
   - Requires gradient clipping for stability

2. **Enable Early Stopping**
   ```python
   {
       "early_stop_metric": "val/f1_score",
       "early_stop_patience": 3,
       "val_check_interval": 0.25
   }
   ```
   - Prevents overfitting
   - Reduces unnecessary training time
   - Automatically saves best checkpoint

3. **Use Appropriate Class Weights**
   ```python
   {
       "class_weights": [1.0, 10.0]  # For imbalanced data
   }
   ```
   - Critical for imbalanced datasets
   - Weight inversely proportional to class frequency
   - Improves minority class performance

4. **Set Reproducible Random Seeds**
   ```python
   {
       "seed": 42  # Not shown in hyperparameters, but recommended
   }
   ```
   - Enables reproducible results
   - Important for debugging and comparison
   - Set in Lightning trainer

5. **Use Checkpoint Resume for Long Training**
   ```python
   {
       "load_ckpt": true,
       "max_epochs": 50
   }
   ```
   - Resume from best checkpoint
   - Handles training interruptions
   - Enables iterative improvement

### For Development

1. **Start with Smaller Models**
   ```python
   {
       "model_class": "bert",  # Simpler baseline
       "max_epochs": 5,
       "batch_size": 16
   }
   ```
   - Faster iteration cycles
   - Easier debugging
   - Establishes performance baseline

2. **Use Validation Check Intervals**
   ```python
   {
       "val_check_interval": 0.25  # Check 4 times per epoch
   }
   ```
   - Earlier detection of training issues
   - Better early stopping behavior
   - More granular metric tracking

3. **Enable Verbose Logging**
   - Check TensorBoard logs regularly
   - Monitor training curves for issues
   - Validate metric computation

4. **Test with Smaller Dataset First**
   - Verify pipeline works end-to-end
   - Debug data issues quickly
   - Estimate resource requirements

### For Performance Optimization

1. **Optimize Batch Size**
   ```python
   {
       "batch_size": 64,  # Larger = faster, but more memory
       "fp16": true
   }
   ```
   - Find largest batch size that fits in memory
   - Use fp16 to fit larger batches
   - Monitor GPU utilization

2. **Use Gradient Accumulation for Large Batches**
   ```python
   {
       "batch_size": 16,
       "accumulate_grad_batches": 4  # Effective batch size = 64
   }
   ```
   - Simulate larger batch sizes
   - Reduces memory requirements
   - Maintains training stability

3. **Optimize Data Loading**
   - Use Parquet format for faster I/O
   - Set num_workers > 0 in DataLoader
   - Cache preprocessed data

4. **Use Distributed Training for Large Datasets**
   ```python
   # Automatically handled by Lightning
   # Just use multiple GPUs
   ```
   - Linear speedup with GPUs
   - FSDP for very large models
   - Handles synchronization automatically

## Example Configurations

### Example 1: Binary Classification with BERT
```python
{
    "id_name": "message_id",
    "text_name": "dialogue_text",
    "label_name": "is_fraud",
    "batch_size": 32,
    "full_field_list": ["message_id", "dialogue_text", "is_fraud", "seller_rating", "price"],
    "cat_field_list": ["marketplace"],
    "tab_field_list": ["seller_rating", "price"],
    "categorical_features_to_encode": ["marketplace"],
    "max_sen_len": 512,
    "max_total_chunks": 5,
    "tokenizer": "bert-base-multilingual-cased",
    "hidden_common_dim": 128,
    "num_classes": 2,
    "is_binary": true,
    "model_class": "multimodal_bert",
    "max_epochs": 10,
    "lr": 2e-5,
    "optimizer": "AdamW",
    "class_weights": [1.0, 5.0],
    "early_stop_metric": "val/f1_score",
    "early_stop_patience": 3,
    "fp16": true,
    "metric_choices": ["auroc", "f1_score", "precision", "recall"]
}
```

**Use Case**: Fraud detection in seller messages with imbalanced classes

### Example 2: Multiclass Classification with CNN
```python
{
    "id_name": "order_id",
    "text_name": "review_text",
    "label_name": "sentiment",
    "batch_size": 64,
    "full_field_list": ["order_id", "review_text", "sentiment", "rating", "verified"],
    "cat_field_list": ["verified"],
    "tab_field_list": ["rating"],
    "categorical_features_to_encode": ["verified"],
    "max_sen_len": 256,
    "kernel_size": [3, 5, 7],
    "num_channels": [128, 128],
    "num_classes": 3,
    "is_binary": false,
    "multiclass_categories": ["positive", "neutral", "negative"],
    "model_class": "multimodal_cnn",
    "max_epochs": 15,
    "lr": 0.001,
    "optimizer": "Adam",
    "dropout_keep": 0.5,
    "early_stop_metric": "val/f1_score",
    "val_check_interval": 0.5,
    "metric_choices": ["auroc", "f1_score", "average_precision"]
}
```

**Use Case**: Sentiment classification on product reviews

### Example 3: Text-Only BERT Fine-tuning
```python
{
    "id_name": "doc_id",
    "text_name": "document",
    "label_name": "category",
    "batch_size": 16,
    "full_field_list": ["doc_id", "document", "category"],
    "cat_field_list": [],
    "tab_field_list": [],
    "max_sen_len": 512,
    "tokenizer": "bert-base-uncased",
    "num_classes": 5,
    "is_binary": false,
    "multiclass_categories": ["tech", "business", "sports", "politics", "entertainment"],
    "model_class": "bert",
    "max_epochs": 20,
    "lr": 3e-5,
    "optimizer": "AdamW",
    "adam_epsilon": 1e-8,
    "warmup_steps": 500,
    "reinit_pooler": true,
    "reinit_layers": 2,
    "gradient_clip_val": 1.0,
    "fp16": true,
    "early_stop_patience": 5
}
```

**Use Case**: Document classification without tabular features

### Example 4: Lightweight LSTM for Fast Inference
```python
{
    "id_name": "ticket_id",
    "text_name": "message",
    "label_name": "priority",
    "batch_size": 128,
    "full_field_list": ["ticket_id", "message", "priority", "response_time"],
    "cat_field_list": [],
    "tab_field_list": ["response_time"],
    "max_sen_len": 256,
    "tokenizer": "bert-base-uncased",
    "num_classes": 2,
    "is_binary": true,
    "model_class": "lstm",
    "num_layers": 2,
    "hidden_common_dim": 64,
    "max_epochs": 25,
    "lr": 0.01,
    "optimizer": "SGD",
    "momentum": 0.9,
    "dropout_keep": 0.6,
    "metric_choices": ["auroc", "f1_score"]
}
```

**Use Case**: Fast, lightweight model for real-time ticket prioritization

### Example 5: Advanced Multimodal with Mixture of Experts
```python
{
    "id_name": "transaction_id",
    "text_name": "description",
    "label_name": "risk_level",
    "batch_size": 32,
    "full_field_list": ["transaction_id", "description", "risk_level", "amount", "frequency", "location"],
    "cat_field_list": ["location"],
    "tab_field_list": ["amount", "frequency"],
    "categorical_features_to_encode": ["location"],
    "max_sen_len": 512,
    "tokenizer": "bert-base-multilingual-cased",
    "hidden_common_dim": 256,
    "num_classes": 4,
    "is_binary": false,
    "multiclass_categories": ["low", "medium", "high", "critical"],
    "model_class": "multimodal_moe",
    "max_epochs": 30,
    "lr": 1e-4,
    "optimizer": "AdamW",
    "class_weights": [1.0, 2.0, 5.0, 10.0],
    "gradient_clip_val": 1.0,
    "fp16": true,
    "early_stop_metric": "val/f1_score",
    "early_stop_patience": 5,
    "val_check_interval": 0.25,
    "warmup_steps": 1000,
    "run_scheduler": true
}
```

**Use Case**: Complex risk assessment with heavy class imbalance

## Integration Patterns

### Upstream Integration (Data Preprocessing)

```
TabularPreprocessing
   ↓ (outputs: preprocessed_train, preprocessed_val, preprocessed_test)
PyTorchTraining
   ↓ (outputs: model.pth, model_artifacts.pth, model.onnx)
```

**Data Flow**:
1. TabularPreprocessing creates train/val/test splits
2. PyTorchTraining loads splits from `/opt/ml/input/data/{train,val,test}/`
3. Preprocessing happens inside training script (tokenization, encoding)

### Downstream Integration (Model Evaluation)

```
PyTorchTraining
   ↓ (outputs: model.pth, model_artifacts.pth, predict_results.pth)
ModelMetricsComputation
   ↓ (outputs: comprehensive_metrics.json, visualizations/)
```

**Artifact Flow**:
1. Training produces model files and prediction results
2. Metrics computation uses predict_results.pth
3. Additional metrics and business-specific analysis

### Deployment Integration (Model Packaging)

```
PyTorchTraining
   ↓ (outputs: model.pth, model_artifacts.pth, model.onnx)
Package
   ↓ (outputs: deployment_package.tar.gz)
Registration
   ↓ (outputs: registered_model_id)
```

**Deployment Flow**:
1. Training produces PyTorch and ONNX models
2. Package step creates MIMS-compatible package
3. Registration step registers in model catalog
4. Model ready for endpoint deployment

### Workflow Example: Complete ML Pipeline

```
1. DummyDataLoading/CradleDataLoading
   ↓ (raw_data)
2. TabularPreprocessing
   ↓ (preprocessed train/val/test)
3. PyTorchTraining
   ↓ (trained model + artifacts)
4. ModelMetricsComputation
   ↓ (evaluation metrics)
5. Package
   ↓ (deployment package)
6. Registration
   ↓ (registered model)
7. Payload
   ↓ (test payloads for endpoint)
```

## Troubleshooting

### Issue 1: Out of Memory Errors

**Symptom**: Training crashes with CUDA out of memory error

**Common Causes**:
1. Batch size too large for available GPU memory
2. Model architecture too large
3. Accumulation of intermediate tensors
4. Multiple processes on same GPU

**Solution**:
```python
# Try these in order:
1. Reduce batch_size (e.g., 64 → 32 → 16)
2. Enable fp16 mixed precision
3. Use gradient accumulation:
   {
       "batch_size": 16,
       "accumulate_grad_batches": 4
   }
4. Use simpler model (multimodal_bert → bert → lstm)
5. Reduce max_sen_len (512 → 256 → 128)
```

### Issue 2: Training Not Converging

**Symptom**: Loss not decreasing or validation metrics not improving

**Common Causes**:
1. Learning rate too high or too low
2. Class imbalance not addressed
3. Insufficient warmup steps (for BERT models)
4. Incorrect data preprocessing

**Solution**:
```python
# For BERT models:
{
    "lr": 2e-5,  # Start with standard BERT lr
    "warmup_steps": 500,
    "run_scheduler": true,
    "gradient_clip_val": 1.0
}

# For imbalanced data:
{
    "class_weights": [1.0, 5.0],  # Balance minority class
    "early_stop_metric": "val/f1_score"  # Focus on F1 over accuracy
}
```

### Issue 3: Multiclass Label Errors

**Symptom**: Errors during label processing or inconsistent predictions

**Common Causes**:
1. `multiclass_categories` doesn't match actual labels in data
2. Missing or extra categories in configuration
3. Unknown labels in test set

**Solution**:
```python
# Verify categories match data:
1. Check unique labels in your data
2. Ensure multiclass_categories list includes all labels
3. Maintain same order as during training
4. Use strict=True to catch unknown labels early

# Example:
{
    "is_binary": false,
    "num_classes": 3,
    "multiclass_categories": ["positive", "neutral", "negative"]
}
```

### Issue 4: ONNX Export Failures

**Symptom**: Model trains successfully but ONNX export fails

**Common Causes**:
1. Custom layers not supported by ONNX
2. Dynamic control flow in model
3. Unsupported PyTorch operations

**Solution**:
- Simplify model architecture if possible
- Check ONNX operator support for PyTorch version
- Use PyTorch (.pth) model for inference if ONNX problematic
- Consider exporting to TorchScript instead

### Issue 5: Slow Training Speed

**Symptom**: Training taking much longer than expected

**Common Causes**:
1. Data loading bottleneck
2. Inefficient batch size
3. Not using available GPUs
4. Text preprocessing overhead

**Solution**:
```python
1. Enable data loader workers (in DataLoader, not hyperparameters)
2. Use Parquet instead of CSV
3. Enable fp16 mixed precision
4. Cache preprocessed data
5. Increase batch size if memory allows
6. Verify GPU utilization with nvidia-smi
```

## References

### Related Scripts

- **Training Scripts:**
  - [`xgboost_training.py`](../../src/cursus/steps/scripts/xgboost_training.py): XGBoost training script (comparison)
  - [`lightgbm_training.py`](../../src/cursus/steps/scripts/lightgbm_training.py): LightGBM training script (comparison)

- **Preprocessing Scripts:**
  - [`tabular_preprocessing.py`](tabular_preprocess_script.md): Upstream data preprocessing
  - [`missing_value_imputation.py`](missing_value_imputation_script.md): Feature imputation
  - [`feature_selection.py`](feature_selection_script.md): Feature selection

- **Evaluation Scripts:**
  - [`model_metrics_computation.py`](model_metrics_computation_script.md): Comprehensive metrics computation
  - [`model_wiki_generator.py`](model_wiki_generator_script.md): Automated documentation

- **Deployment Scripts:**
  - [`package.py`](package_script.md): Model packaging for MIMS
  - [`payload.py`](payload_script.md): Test payload generation

### Related Documentation

- **Contract**: [`src/cursus/steps/contracts/pytorch_training_contract.py`](../../src/cursus/steps/contracts/pytorch_training_contract.py) - Complete contract specification
- **Config**: [`src/cursus/steps/configs/config_pytorch_training_step.py`](../../src/cursus/steps/configs/config_pytorch_training_step.py) - Configuration class
- **Builder**: [`src/cursus/steps/builders/builder_pytorch_training_step.py`](../../src/cursus/steps/builders/builder_pytorch_training_step.py) - Step builder
- **Specification**: PyTorch Training step specification in registry

### Related Design Documents

No specific design documents currently exist for this training script. General training step patterns are documented in:

- **[Training Step Builder Patterns](../1_design/training_step_builder_patterns.md)**: Common patterns for training step implementation
- **[Training Step Alignment Validation Patterns](../1_design/training_step_alignment_validation_patterns.md)**: Validation patterns for training steps
- **[Registry Based Step Name Generation](../1_design/registry_based_step_name_generation.md)**: Step naming and registration patterns

### External References

- **[PyTorch Documentation](https://pytorch.org/docs/stable/index.html)**: Official PyTorch framework documentation
- **[PyTorch Lightning Documentation](https://pytorch-lightning.readthedocs.io/)**: Lightning framework for training orchestration
- **[HuggingFace Transformers](https://huggingface.co/docs/transformers/)**: BERT and transformer models
- **[ONNX Documentation](https://onnx.ai/onnx/)**: ONNX model format specification
- **[TorchMetrics](https://torchmetrics.readthedocs.io/)**: Metrics computation library
- **[FSDP Documentation](https://pytorch.org/docs/stable/fsdp.html)**: Fully Sharded Data Parallel training guide
- **[Mixed Precision Training](https://pytorch.org/docs/stable/amp.html)**: Automatic mixed precision guide

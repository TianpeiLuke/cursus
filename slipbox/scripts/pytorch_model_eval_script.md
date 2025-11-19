---
tags:
  - code
  - processing_script
  - model_evaluation
  - pytorch
  - deep_learning
keywords:
  - pytorch model evaluation
  - lightning framework
  - GPU acceleration
  - multi-modal models
  - bimodal bert
  - trimodal bert
  - format preservation
  - ROC curves
  - PR curves
topics:
  - model evaluation
  - deep learning inference
  - PyTorch Lightning
  - multi-modal learning
language: python
date of note: 2025-11-19
---

# PyTorch Model Evaluation Script

## Overview

The `pytorch_model_eval.py` script evaluates trained PyTorch Lightning models with comprehensive metrics computation, visualization generation, and GPU/CPU support for deep learning models in production ML pipelines.

The script provides a production-ready evaluation workflow that supports multi-modal architectures (text, tabular, bimodal, trimodal), automatic GPU/CPU detection, format preservation across data formats, and comprehensive performance metrics using PyTorch Lightning utilities. It integrates seamlessly with the Cursus pipeline framework following the same contract structure as XGBoost evaluation.

Key capabilities include automatic device detection and explicit GPU/CPU control, multi-modal model support (bimodal BERT, trimodal BERT with dual text, trimodal cross-attention, trimodal gate fusion), format preservation (CSV/TSV/Parquet), comprehensive preprocessing pipeline (text tokenization, categorical encoding, label processing), PyTorch Lightning inference utilities, ROC and PR curve generation, and comparison mode support for A/B testing.

## Purpose and Major Tasks

### Primary Purpose
Evaluate trained PyTorch Lightning models on held-out test data, generating comprehensive performance metrics, visualizations, and predictions for model validation and production monitoring.

### Major Tasks

1. **Model Artifact Loading**: Load trained PyTorch Lightning model with all preprocessing artifacts (config, embeddings, vocab, processors)
2. **Data Loading**: Load evaluation data with automatic format detection (CSV/TSV/Parquet)
3. **Preprocessing Pipeline**: Apply complete text, categorical, and label preprocessing matching training
4. **Device Setup**: Configure GPU/CPU environment with automatic detection and optimization
5. **Inference Execution**: Generate predictions using PyTorch Lightning inference utilities with batch processing
6. **Metrics Computation**: Compute comprehensive evaluation metrics (AUC-ROC, Average Precision, F1, etc.)
7. **Visualization Generation**: Create ROC and PR curves using Lightning plotting utilities
8. **Prediction Export**: Save predictions with format preservation and provenance tracking
9. **Metrics Export**: Save metrics as JSON and human-readable summary
10. **Health Monitoring**: Create success/failure markers for pipeline orchestration

## Script Contract

### Entry Point
```
pytorch_model_eval.py
```

### Input Paths

| Path | Location | Description |
|------|----------|-------------|
| `model_input` | `/opt/ml/processing/input/model` | Trained PyTorch model artifacts directory |
| `processed_data` | `/opt/ml/processing/input/eval_data` | Evaluation data directory (CSV/TSV/Parquet) |

**Model Artifacts Expected**:
```
/opt/ml/processing/input/model/
├── model.pth                    # Trained PyTorch Lightning checkpoint
├── model_artifacts.pth          # Config, embeddings, vocab, processors
└── hyperparameters.json         # Model configuration and hyperparameters
```

**Evaluation Data Expected**:
```
/opt/ml/processing/input/eval_data/
├── eval_data.csv (or .tsv, .parquet)
└── _SUCCESS (optional marker)
```

### Output Paths

| Path | Location | Description |
|------|----------|-------------|
| `eval_output` | `/opt/ml/processing/output/eval` | Evaluation predictions with probabilities |
| `metrics_output` | `/opt/ml/processing/output/metrics` | Comprehensive metrics and visualizations |

**Evaluation Output Contents**:
```
/opt/ml/processing/output/eval/
└── eval_predictions.{csv,tsv,parquet}  # ID, true label, class probabilities
```

**Metrics Output Contents**:
```
/opt/ml/processing/output/metrics/
├── metrics.json                 # All metrics in JSON format
├── metrics_summary.txt          # Human-readable metrics summary
├── roc_curve.jpg               # ROC curve visualization
├── pr_curve.jpg                # Precision-Recall curve visualization
├── tensorboard_eval/           # TensorBoard logs for plots
├── _SUCCESS                    # Success marker file
└── _HEALTH                     # Health check file with timestamp
```

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ID_FIELD` | Column name for record IDs | `"order_id"` |
| `LABEL_FIELD` | Column name for true labels | `"label"` |

### Optional Environment Variables

#### Device Configuration
| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `"auto"` | Device selection: "auto", "cuda", "cpu" |
| `ACCELERATOR` | `"auto"` | PyTorch Lightning accelerator: "auto", "gpu", "cpu" |

#### Performance Tuning
| Variable | Default | Description |
|----------|---------|-------------|
| `BATCH_SIZE` | `"32"` | Batch size for inference |
| `NUM_WORKERS` | `"0"` | Number of workers for data loading |

#### Comparison Mode (Planned Feature)
| Variable | Default | Description | Status |
|----------|---------|-------------|--------|
| `COMPARISON_MODE` | `"false"` | Enable comparison with baseline | Placeholder |
| `PREVIOUS_SCORE_FIELD` | `""` | Field with previous model scores | Placeholder |
| `COMPARISON_METRICS` | `"all"` | Metrics to compare | Placeholder |
| `STATISTICAL_TESTS` | `"true"` | Perform statistical significance tests | Placeholder |
| `COMPARISON_PLOTS` | `"true"` | Generate comparison plots | Placeholder |

### Job Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--job_type` | `str` | Yes | Job type identifier (e.g., "testing", "validation") |

**Usage Example**:
```bash
python pytorch_model_eval.py --job_type testing
```

## Input Data Structure

### Expected Input Format

```
/opt/ml/processing/input/eval_data/
├── evaluation_data.csv (or .tsv, .parquet)
└── _SUCCESS (optional marker)
```

### Required Columns

The evaluation data must contain:

- **ID Column**: Unique identifier for each record (configurable via `ID_FIELD`)
  - Default name: `"order_id"`
  - Type: Any unique identifier (string, integer)

- **Label Column**: True labels for evaluation (configurable via `LABEL_FIELD`)
  - Default name: `"label"`
  - Type: Binary (0/1) or multiclass (0, 1, 2, ...)

- **Text Fields**: Depends on model configuration
  - For bimodal models: Single text field (e.g., `"text"`, `"dialogue"`)
  - For trimodal models: Two text fields (e.g., `"primary_text"`, `"secondary_text"`)

- **Categorical Fields**: As defined in model configuration
  - Examples: `"category"`, `"marketplace"`, `"seller_type"`
  - Processed using learned categorical mappings from training

### Supported Data Formats

1. **CSV Files**: Comma-separated values
2. **TSV Files**: Tab-separated values
3. **Parquet Files**: Columnar format for efficient storage

**Format Preservation**:
- Script automatically detects input format
- All outputs use the same format as input
- Supports nested directories with recursive search

### Data Requirements

**Text Data**:
- UTF-8 encoding
- May contain HTML tags (will be normalized)
- May contain emojis (will be removed)
- Dialogue format supported with speaker/message structure

**Categorical Data**:
- Must match training vocabulary
- Unknown categories handled gracefully (mapped to default)
- Missing values filled with mode or default

**Label Data**:
- Binary: 0 or 1
- Multiclass: Consecutive integers starting from 0
- Must match training label space

## Output Data Structure

### Evaluation Predictions Output

```
/opt/ml/processing/output/eval/
└── eval_predictions.{csv,tsv,parquet}
```

**Columns in Output**:
- `{ID_FIELD}`: Original record ID (e.g., `order_id`)
- `{LABEL_FIELD}`: True label value
- `prob_class_0`: Predicted probability for class 0
- `prob_class_1`: Predicted probability for class 1
- `prob_class_N`: Additional columns for multiclass (N classes total)

**Example Output Structure** (Binary Classification):
```csv
order_id,label,prob_class_0,prob_class_1
12345,0,0.8234,0.1766
12346,1,0.2156,0.7844
12347,0,0.9012,0.0988
```

**Example Output Structure** (Multiclass):
```csv
order_id,label,prob_class_0,prob_class_1,prob_class_2
12345,0,0.7234,0.1766,0.1000
12346,2,0.1156,0.2844,0.6000
```

### Metrics Output Directory

```
/opt/ml/processing/output/metrics/
├── metrics.json                 # Machine-readable metrics
├── metrics_summary.txt          # Human-readable summary
├── roc_curve.jpg               # ROC curve plot
├── pr_curve.jpg                # Precision-Recall curve plot
├── tensorboard_eval/           # TensorBoard logs
│   └── events.out.tfevents.*
├── _SUCCESS                    # Pipeline success marker
└── _HEALTH                     # Health check with timestamp
```

**metrics.json Contents**:
```json
{
  "eval/auroc": 0.8542,
  "eval/average_precision": 0.8012,
  "eval/f1_score": 0.7823,
  "eval/precision": 0.7654,
  "eval/recall": 0.8001,
  "eval/accuracy": 0.8234
}
```

**metrics_summary.txt Contents**:
```
PYTORCH MODEL EVALUATION METRICS
==================================================

AUC-ROC:           0.8542
Average Precision: 0.8012
F1 Score:          0.7823

==================================================

ALL METRICS
==================================================
eval/accuracy: 0.823400
eval/auroc: 0.854200
eval/average_precision: 0.801200
eval/f1_score: 0.782300
eval/precision: 0.765400
eval/recall: 0.800100
```

## Key Functions and Tasks

### File I/O with Format Preservation

#### `_detect_file_format(file_path)`
**Purpose**: Detect data format from file extension

**Algorithm**:
```python
1. Extract file suffix (.csv, .tsv, .parquet)
2. Map suffix to format string
3. Raise error if unsupported format
4. Return format string
```

**Parameters**:
- `file_path` (Path): Path to the file

**Returns**: `str` - Format string ('csv', 'tsv', 'parquet')

#### `load_dataframe_with_format(file_path)`
**Purpose**: Load DataFrame and detect its format

**Algorithm**:
```python
1. Detect format from file extension
2. Load DataFrame based on format:
   - CSV: pd.read_csv()
   - TSV: pd.read_csv(sep='\t')
   - Parquet: pd.read_parquet()
3. Return (DataFrame, format_string)
```

**Parameters**:
- `file_path` (Path): Path to the file

**Returns**: `Tuple[pd.DataFrame, str]` - (DataFrame, format)

#### `save_dataframe_with_format(df, output_path, format_str)`
**Purpose**: Save DataFrame preserving original format

**Algorithm**:
```python
1. Determine output file path with correct extension
2. Save based on format:
   - CSV: df.to_csv()
   - TSV: df.to_csv(sep='\t')
   - Parquet: df.to_parquet()
3. Return saved file path
```

**Parameters**:
- `df` (pd.DataFrame): DataFrame to save
- `output_path` (Path): Base output path
- `format_str` (str): Format to save in

**Returns**: `Path` - Path to saved file

### Model Artifact Loading

#### `load_model_artifacts(model_dir)`
**Purpose**: Load trained PyTorch model and all preprocessing artifacts

**Algorithm**:
```python
1. Load hyperparameters.json from model_dir
2. Load model_artifacts.pth:
   - Extract config dictionary
   - Extract embedding matrix (if applicable)
   - Extract vocabulary mappings
3. Reconstruct tokenizer from config:
   - Default: bert-base-multilingual-cased
   - Load from Hugging Face transformers
4. Load trained model weights from model.pth:
   - Use model_class from config
   - Load to CPU for evaluation
   - Set to eval() mode
5. Extract preprocessing processors:
   - Categorical processor mappings
   - Label ID mappings (label_to_id, id_to_label)
6. Package and return all artifacts
```

**Parameters**:
- `model_dir` (str): Path to model artifacts directory

**Returns**: `Tuple[nn.Module, Dict, AutoTokenizer, Dict]` - (model, config, tokenizer, processors)

**Example Usage**:
```python
model, config, tokenizer, processors = load_model_artifacts("/opt/ml/processing/input/model")
# model: PyTorch Lightning model in eval mode
# config: {'model_class': 'bimodal_bert', 'num_classes': 2, ...}
# tokenizer: BertTokenizer
# processors: {'categorical_processors': {...}, 'label_mappings': {...}}
```

### Data Preprocessing Pipeline

#### `preprocess_eval_data(df, config, tokenizer, processors, eval_data_dir, filename)`
**Purpose**: Apply complete preprocessing pipeline matching training

**Algorithm**:
```python
1. Create BSMDataset from file
2. Fill missing values in categorical and label columns
3. Build text preprocessing pipeline:
   a. DialogueSplitterProcessor - Split speaker/message format
   b. HTMLNormalizerProcessor - Remove HTML tags
   c. EmojiRemoverProcessor - Remove emojis
   d. TextNormalizationProcessor - Normalize whitespace/punctuation
   e. DialogueChunkerProcessor - Chunk long dialogues with tokenizer
   f. BertTokenizeProcessor - Tokenize with BERT, create input_ids & attention_mask
4. Add text pipeline to dataset
5. Add categorical processors:
   - For each categorical field with learned mapping
   - Apply CategoricalLabelProcessor with category_to_label mapping
6. Add label processor for multiclass:
   - If not binary and num_classes > 2
   - Apply MultiClassLabelProcessor with label_to_id mapping
7. Determine model modality:
   - Check if trimodal model (trimodal_bert, trimodal_cross_attn_bert, trimodal_gate_fusion_bert)
   - Check if dual text config exists (primary_text_name, secondary_text_name)
8. Select appropriate collate function:
   - Trimodal: build_trimodal_collate_batch (handles dual text modalities)
   - Bimodal: build_collate_batch (handles single text modality)
9. Create DataLoader with collate function and batch size
10. Return BSMDataset and DataLoader
```

**Parameters**:
- `df` (pd.DataFrame): Input DataFrame
- `config` (Dict): Model configuration
- `tokenizer` (AutoTokenizer): BERT tokenizer
- `processors` (Dict): Preprocessing processors
- `eval_data_dir` (str): Data directory
- `filename` (str): Data filename

**Returns**: `Tuple[BSMDataset, DataLoader]` - (dataset, dataloader)

**Preprocessing Pipeline Stages**:
1. **Text Processing**: HTML normalization → Emoji removal → Text normalization → Dialogue chunking → BERT tokenization
2. **Categorical Processing**: Category-to-label mapping using learned vocabulary
3. **Label Processing**: Label-to-ID mapping for multiclass tasks

### Device Setup and GPU Optimization

#### `setup_device_environment(device)`
**Purpose**: Configure GPU/CPU environment with automatic detection and optimization

**Algorithm**:
```python
1. Determine device:
   - If device="auto":
     - Check torch.cuda.is_available()
     - Use "cuda" if available, else "cpu"
   - Else: Use specified device
2. Set accelerator:
   - "gpu" if device="cuda"
   - "cpu" otherwise
3. If using GPU:
   a. Log GPU name and count
   b. Enable cudnn.benchmark for optimization
   c. Log GPU memory (allocated, reserved)
4. Return (device_string, accelerator_string)
```

**Parameters**:
- `device` (str): Device selection ("auto", "cuda", "cpu")

**Returns**: `Tuple[str, str]` - (device, accelerator)

**GPU Optimization Features**:
- `torch.backends.cudnn.benchmark = True`: Enables cuDNN auto-tuner for optimal conv algorithms
- Memory logging: Tracks GPU memory usage
- Automatic device selection: Falls back to CPU if GPU unavailable

**Example Output**:
```
Using device: cuda, accelerator: gpu
GPU: NVIDIA A100-SXM4-40GB
GPU Count: 1
GPU Memory Allocated: 0.02 GB
GPU Memory Reserved: 0.08 GB
```

### Prediction Generation

#### `generate_predictions(model, dataloader, device, accelerator)`
**Purpose**: Generate predictions using PyTorch Lightning inference utilities

**Algorithm**:
```python
1. Log inference start with device and accelerator
2. Call Lightning's model_inference utility:
   - Pass model and dataloader
   - Specify accelerator type (gpu/cpu)
   - Specify device string
   - No logging during evaluation (model_log_path=None)
3. Receive predictions and true labels:
   - y_pred: Probability predictions (n_samples, n_classes)
   - y_true: True labels (n_samples,)
4. Log shapes for verification
5. Return (y_pred, y_true)
```

**Parameters**:
- `model` (nn.Module): PyTorch Lightning model
- `dataloader` (DataLoader): Evaluation data loader
- `device` (str): Device string
- `accelerator` (str): Accelerator type

**Returns**: `Tuple[np.ndarray, np.ndarray]` - (y_pred probabilities, y_true labels)

**Batch Processing**:
- Processes data in batches for memory efficiency
- Automatic gradient computation disabled (eval mode)
- Mixed precision support via PyTorch Lightning

### Metrics Computation

#### `compute_evaluation_metrics(y_true, y_prob, config)`
**Purpose**: Compute comprehensive evaluation metrics using Lightning utilities

**Algorithm**:
```python
1. Determine task type:
   - "binary" if config["is_binary"] is True
   - "multiclass" otherwise
2. Extract num_classes from config
3. Define metrics to compute:
   - "auroc": Area Under ROC Curve
   - "average_precision": Average Precision (AP)
   - "f1_score": F1 Score
4. Call Lightning's compute_metrics:
   - Pass y_prob, y_true
   - Pass output_metrics list
   - Specify task and num_classes
   - Set stage="eval"
5. Receive metrics dictionary with "eval/" prefix
6. Log formatted metrics summary
7. Return metrics dictionary
```

**Parameters**:
- `y_true` (np.ndarray): True labels
- `y_prob` (np.ndarray): Predicted probabilities
- `config` (Dict): Model configuration

**Returns**: `Dict[str, float]` - Metrics dictionary

**Metrics Computed**:

**Binary Classification**:
- `eval/auroc`: AUC-ROC score (0-1, higher is better)
- `eval/average_precision`: Average Precision (0-1, higher is better)
- `eval/f1_score`: F1 Score (0-1, higher is better)
- `eval/precision`: Precision (0-1, higher is better)
- `eval/recall`: Recall (0-1, higher is better)
- `eval/accuracy`: Accuracy (0-1, higher is better)

**Multiclass Classification**:
- `eval/auroc_macro`: Macro-averaged AUC-ROC
- `eval/auroc_micro`: Micro-averaged AUC-ROC
- `eval/average_precision_macro`: Macro-averaged AP
- `eval/f1_score_macro`: Macro-averaged F1

#### `log_metrics_summary(metrics, is_binary)`
**Purpose**: Log formatted metrics summary with key highlights

**Algorithm**:
```python
1. Get current timestamp
2. Log header with separator
3. For each metric in dictionary:
   a. Format numeric values to 4 decimal places
   b. Log with consistent spacing: "METRIC: name = value"
4. Log key performance metrics section:
   - Binary: AUC-ROC, Average Precision, F1 Score
   - Multiclass: Macro AUC-ROC, Micro AUC-ROC
5. Log footer separator
```

**Parameters**:
- `metrics` (Dict): Metrics dictionary
- `is_binary` (bool): Whether binary classification

**Returns**: None (logs to console)

### Visualization Generation

#### `generate_evaluation_plots(y_true, y_prob, config, output_dir)`
**Purpose**: Generate ROC and PR curves using Lightning plotting utilities

**Algorithm**:
```python
1. Determine task type and num_classes from config
2. Create TensorBoard writer:
   - Log directory: {output_dir}/tensorboard_eval
3. Generate ROC curves:
   a. Call roc_metric_plot from Lightning utilities
   b. Pass y_pred, y_true, validation copies
   c. Pass task type and num_classes
   d. Pass TensorBoard writer and global_step=0
   e. Save to {output_dir}/roc_curve.jpg
4. Generate PR curves:
   a. Call pr_metric_plot from Lightning utilities
   b. Pass same parameters as ROC
   c. Save to {output_dir}/pr_curve.jpg
5. Close TensorBoard writer
6. Return dictionary of plot paths
```

**Parameters**:
- `y_true` (np.ndarray): True labels
- `y_prob` (np.ndarray): Predicted probabilities
- `config` (Dict): Model configuration
- `output_dir` (str): Output directory

**Returns**: `Dict[str, str]` - Plot paths dictionary

**Generated Plots**:
1. **ROC Curve**: True Positive Rate vs False Positive Rate
   - Binary: Single curve with AUC score
   - Multiclass: One-vs-rest curves for each class

2. **PR Curve**: Precision vs Recall
   - Binary: Single curve with Average Precision
   - Multiclass: Per-class curves

## Algorithms and Data Structures

### Multi-Modal Data Collation

**Problem**: Batch together variable-length text sequences with tabular features for multi-modal models

**Solution Strategy**:
1. Pad text sequences to maximum length in batch
2. Create attention masks for valid tokens
3. Stack tabular features as tensors
4. Handle both bimodal (single text) and trimodal (dual text) architectures

**Bimodal Collate Algorithm**:
```python
def build_collate_batch(input_ids_key, attention_mask_key):
    def collate_batch(batch):
        # Extract components from batch
        text_input_ids = [item[input_ids_key] for item in batch]
        attention_masks = [item[attention_mask_key] for item in batch]
        tabular_features = [item['tabular'] for item in batch]
        labels = [item['label'] for item in batch]
        
        # Pad text sequences
        max_len = max(len(ids) for ids in text_input_ids)
        padded_ids = pad_sequence(text_input_ids, max_len, pad_value=0)
        padded_masks = pad_sequence(attention_masks, max_len, pad_value=0)
        
        # Stack tabular features and labels
        tabular_tensor = torch.stack(tabular_features)
        label_tensor = torch.tensor(labels)
        
        return {
            'input_ids': padded_ids,
            'attention_mask': padded_masks,
            'tabular': tabular_tensor,
            'labels': label_tensor
        }
    
    return collate_batch
```

**Trimodal Collate Algorithm**:
```python
def build_trimodal_collate_batch(
    primary_input_ids_key, primary_attention_mask_key,
    secondary_input_ids_key, secondary_attention_mask_key
):
    def collate_batch(batch):
        # Extract primary text modality
        primary_ids = [item[primary_input_ids_key] for item in batch]
        primary_masks = [item[primary_attention_mask_key] for item in batch]
        
        # Extract secondary text modality
        secondary_ids = [item[secondary_input_ids_key] for item in batch]
        secondary_masks = [item[secondary_attention_mask_key] for item in batch]
        
        # Extract tabular and labels
        tabular_features = [item['tabular'] for item in batch]
        labels = [item['label'] for item in batch]
        
        # Pad both text modalities independently
        primary_padded = pad_sequence(primary_ids, ...)
        primary_masks_padded = pad_sequence(primary_masks, ...)
        secondary_padded = pad_sequence(secondary_ids, ...)
        secondary_masks_padded = pad_sequence(secondary_masks, ...)
        
        # Stack tabular and labels
        tabular_tensor = torch.stack(tabular_features)
        label_tensor = torch.tensor(labels)
        
        return {
            'primary_input_ids': primary_padded,
            'primary_attention_mask': primary_masks_padded,
            'secondary_input_ids': secondary_padded,
            'secondary_attention_mask': secondary_masks_padded,
            'tabular': tabular_tensor,
            'labels': label_tensor
        }
    
    return collate_batch
```

**Complexity**:
- Time: O(n * m) where n=batch_size, m=max_sequence_length
- Space: O(n * m) for padded sequences

**Key Features**:
- Dynamic padding per batch (memory efficient)
- Separate handling of dual text modalities for trimodal models
- Attention mask creation for variable-length sequences
- Preserves tabular features without modification

### Text Preprocessing Pipeline

**Problem**: Transform raw dialogue text into BERT-compatible token sequences

**Solution Strategy**: Compose multiple processors in a pipeline pattern

**Algorithm**:
```python
text_pipeline = (
    DialogueSplitterProcessor()           # Step 1
    >> HTMLNormalizerProcessor()          # Step 2
    >> EmojiRemoverProcessor()            # Step 3
    >> TextNormalizationProcessor()       # Step 4
    >> DialogueChunkerProcessor(...)      # Step 5
    >> BertTokenizeProcessor(...)         # Step 6
)

# Pipeline composition operator (>>)
class Processor:
    def __rshift__(self, other):
        return ComposedProcessor(self, other)
    
class ComposedProcessor:
    def __init__(self, first, second):
        self.first = first
        self.second = second
    
    def __call__(self, text):
        intermediate = self.first(text)
        return self.second(intermediate)
```

**Processing Stages**:

1. **DialogueSplitterProcessor**: Parse "Speaker: Message" format
   - Input: "Buyer: Hello\nSeller: Hi there"
   - Output: [{"speaker": "Buyer", "message": "Hello"}, ...]

2. **HTMLNormalizerProcessor**: Remove HTML tags
   - Input: "Hello <b>world</b>"
   - Output: "Hello world"

3. **EmojiRemoverProcessor**: Remove emoji characters
   - Input: "Great! 😊"
   - Output: "Great!"

4. **TextNormalizationProcessor**: Normalize whitespace/punctuation
   - Input: "Hello   world\n\n"
   - Output: "Hello world"

5. **DialogueChunkerProcessor**: Chunk long dialogues
   - Respects max_tokens limit
   - Creates multiple chunks if needed
   - Max total chunks configurable

6. **BertTokenizeProcessor**: BERT tokenization
   - Creates input_ids tensor
   - Creates attention_mask tensor
   - Adds special tokens ([CLS], [SEP])
   - Handles truncation and padding

**Complexity**: O(n) where n = text length

**Benefits**:
- Modular design (easy to add/remove stages)
- Reusable processors
- Type-safe composition
- Matches training preprocessing exactly

## Performance Characteristics

### Inference Performance

| Dataset Size | Device | Batch Size | Inference Time | Memory Usage | Notes |
|--------------|--------|------------|----------------|--------------|-------|
| 1K samples | CPU | 32 | ~30s | ~2GB | Baseline |
| 1K samples | GPU | 32 | ~5s | ~4GB | 6x speedup |
| 10K samples | CPU | 32 | ~5min | ~2GB | Linear scaling |
| 10K samples | GPU | 32 | ~45s | ~5GB | 6-7x speedup |
| 100K samples | GPU | 64 | ~7min | ~8GB | Larger batches help |

**GPU Optimization Impact**:
- `cudnn.benchmark=True`: 10-15% speedup on conv operations
- Batch size 64 vs 32: ~30% throughput improvement
- Mixed precision (planned): 40-50% speedup potential

### Preprocessing Performance

| Operation | Samples/sec | Bottleneck |
|-----------|-------------|------------|
| File I/O (CSV) | 50K | Disk I/O |
| File I/O (Parquet) | 200K | Disk I/O |
| Text Tokenization | 2K | CPU bound |
| Categorical Encoding | 100K | Memory lookup |
| Overall Pipeline | 1.5-2K | Tokenization |

**Optimization Opportunities**:
1. Use Parquet for large datasets (4x faster I/O)
2. Increase batch size on GPUs with sufficient memory
3. Use multiple data loading workers (NUM_WORKERS > 0)
4. Pre-tokenize data if evaluating multiple times

## Error Handling

### Input Validation Errors

**Missing Required Columns**:
```
KeyError: 'order_id' not found in evaluation data
```
**Cause**: ID_FIELD or LABEL_FIELD not in DataFrame

**Resolution**: Verify column names match environment variables, check data file schema

**Invalid Data Format**:
```
RuntimeError: Unsupported file format: .xlsx
```
**Cause**: File format not supported (only CSV/TSV/Parquet)

**Resolution**: Convert to supported format or update script

### Model Loading Errors

**Missing Model Artifacts**:
```
FileNotFoundError: [Errno 2] No such file or directory: '/opt/ml/processing/input/model/model.pth'
```
**Cause**: Model artifacts not present in input directory

**Resolution**: Check model input path, verify training step outputs

**Incompatible Model Architecture**:
```
RuntimeError: Error loading model: model class 'invalid_model' not recognized
```
**Cause**: Model class not supported or corrupted config

**Resolution**: Check hyperparameters.json, verify model_class value

### Device Errors

**CUDA Out of Memory**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```
**Cause**: Batch size too large for GPU memory

**Resolution**:
1. Reduce BATCH_SIZE environment variable
2. Use smaller model
3. Fall back to CPU: `export DEVICE=cpu`

**CUDA Not Available**:
```
Warning: CUDA requested but not available, falling back to CPU
```
**Cause**: GPU requested but not available

**Resolution**: Script automatically falls back to CPU, no action needed

### Processing Errors

**Tokenization Errors**:
```
ValueError: Text length exceeds maximum sequence length
```
**Cause**: Text too long for BERT tokenizer

**Resolution**: Text automatically truncated, check DialogueChunkerProcessor settings

**Label Mismatch**:
```
ValueError: Found label value 3 but model trained with 2 classes
```
**Cause**: Evaluation data has labels not in training set

**Resolution**: Verify label space consistency, check data preprocessing

## Best Practices

### For Production Deployments

1. **Use GPU for Large Datasets**
   ```bash
   export DEVICE=auto  # Automatically uses GPU if available
   export BATCH_SIZE=64  # Larger batches on GPU
   ```
   - GPU provides 6-7x speedup for large datasets
   - Auto mode safely falls back to CPU if needed

2. **Use Parquet Format**
   ```bash
   # Convert CSV to Parquet before evaluation
   # 4x faster I/O, smaller storage
   ```
   - Significantly faster for datasets > 10K samples
   - Better compression and columnar storage

3. **Monitor GPU Memory**
   - Script logs GPU memory usage
   - Start with batch_size=32, increase if memory allows
   - Watch for "CUDA out of memory" errors

4. **Verify Format Preservation**
   - Check output format matches input
   - Verify all columns preserved correctly
   - Validate ID column integrity

### For Development

1. **Start with CPU for Small Datasets**
   ```bash
   export DEVICE=cpu
   export BATCH_SIZE=16
   ```
   - Simpler debugging without GPU complications
   - Faster startup time

2. **Test with Sample Data First**
   - Run on 100-1000 samples before full dataset
   - Verify preprocessing works correctly
   - Check metrics make sense

3. **Use Meaningful Job Types**
   ```bash
   python pytorch_model_eval.py --job_type validation
   python pytorch_model_eval.py --job_type testing
   python pytorch_model_eval.py --job_type production_monitoring
   ```
   - Helps track evaluation purpose in logs

### For Model Validation

1. **Compare Multiple Checkpoints**
   - Evaluate different training checkpoints
   - Track metrics over training iterations
   - Select best checkpoint for deployment

2. **Cross-Validate on Multiple Test Sets**
   - Test on different data splits
   - Verify model generalizes
   - Check for overfitting signs

3. **Monitor Key Metrics**
   - AUC-ROC: Overall discrimination ability
   - Average Precision: Performance on imbalanced data
   - F1 Score: Balance of precision and recall

## Example Configurations

### Example 1: Basic CPU Evaluation
```bash
# Environment variables
export ID_FIELD="order_id"
export LABEL_FIELD="label"
export DEVICE="cpu"
export BATCH_SIZE="32"

# Run evaluation
python pytorch_model_eval.py --job_type testing
```

**Use Case**: Small dataset (< 10K samples), development/testing environment

### Example 2: GPU Evaluation with Auto-Detection
```bash
# Environment variables
export ID_FIELD="transaction_id"
export LABEL_FIELD="is_fraud"
export DEVICE="auto"  # Uses GPU if available
export BATCH_SIZE="64"
export NUM_WORKERS="4"

# Run evaluation
python pytorch_model_eval.py --job_type production_validation
```

**Use Case**: Large dataset, production validation with GPU acceleration

### Example 3: Explicit GPU with Memory Optimization
```bash
# Environment variables
export ID_FIELD="request_id"
export LABEL_FIELD="category"
export DEVICE="cuda"
export ACCELERATOR="gpu"
export BATCH_SIZE="128"  # Large batch for throughput

# Run evaluation
python pytorch_model_eval.py --job_type performance_benchmark
```

**Use Case**: Maximum GPU utilization, performance benchmarking

### Example 4: Multiclass Evaluation
```bash
# Environment variables
export ID_FIELD="sample_id"
export LABEL_FIELD="category"  # Values: 0, 1, 2, 3, 4
export DEVICE="auto"
export BATCH_SIZE="32"

# Run evaluation
python pytorch_model_eval.py --job_type multiclass_validation
```

**Use Case**: Multiclass classification with 5 categories

## Integration Patterns

### Upstream Integration

```
PyTorchTraining
   ↓ (outputs: model.pth, model_artifacts.pth, hyperparameters.json)
PyTorchModelEval
   ↓ (outputs: eval_predictions.{csv,tsv,parquet}, metrics.json, visualizations)
```

### Downstream Integration

```
PyTorchModelEval
   ↓ (outputs: eval_predictions with probabilities)
ModelMetricsComputation (optional - for additional metrics)
   ↓ (outputs: comprehensive metrics report)
Package (for deployment)
   ↓ (outputs: deployable model package)
```

### Complete Multi-Modal Pipeline Example

```
1. DummyDataLoading/CradleDataLoading
   ↓ (multi-modal data: text + tabular)
2. TabularPreprocessing
   ↓ (cleaned and split data)
3. PyTorchTraining
   ↓ (trained bimodal/trimodal BERT model)
4. PyTorchModelEval
   ↓ (evaluation metrics and predictions)
5. ModelMetricsComputation
   ↓ (comprehensive analysis)
6. Package
   ↓ (MIMS deployment package)
```

### Multi-Stage Evaluation Workflow

```
1. Training Phase:
   - PyTorchTraining → checkpoint_epoch_10.pth
   
2. Validation Phase:
   - PyTorchModelEval (job_type=validation)
   - Evaluate on validation set
   - Track metrics over epochs
   
3. Test Phase:
   - PyTorchModelEval (job_type=testing)
   - Final evaluation on held-out test set
   - Generate ROC/PR curves
   
4. Production Monitoring:
   - PyTorchModelEval (job_type=production)
   - Periodic re-evaluation on new data
   - Track metric drift
```

## Troubleshooting

### Issue 1: Slow Inference on GPU

**Symptom**: GPU evaluation not much faster than CPU

**Common Causes**:
1. **Batch size too small**: GPU underutilized
2. **Data loading bottleneck**: CPU can't keep up
3. **Memory transfer overhead**: Small batches cause frequent transfers

**Solution**:
```bash
# Increase batch size
export BATCH_SIZE="128"

# Enable multiple data loading workers
export NUM_WORKERS="4"

# Verify GPU utilization
nvidia-smi  # Should show high GPU usage
```

### Issue 2: Inconsistent Metrics with Training

**Symptom**: Evaluation metrics differ significantly from training metrics

**Common Causes**:
1. **Preprocessing mismatch**: Evaluation preprocessing differs from training
2. **Data distribution shift**: Test data different from training
3. **Model not in eval mode**: Dropout/BatchNorm still active

**Solution**:
- Verify preprocessing pipeline matches training
- Check model.eval() is called (automatic in script)
- Compare data distributions between train and eval
- Review model_artifacts.pth for correct processors

### Issue 3: Missing Predictions for Some Samples

**Symptom**: Output has fewer rows than input

**Common Causes**:
1. **Preprocessing failures**: Some samples fail validation
2. **Memory errors**: Out of memory during batch processing
3. **Label filtering**: Some labels dropped

**Solution**:
- Check logs for preprocessing warnings
- Reduce batch size if memory issues
- Verify all samples have valid text and features
- Check for null values in required columns

### Issue 4: Poor Performance on Trimodal Models

**Symptom**: Trimodal model metrics worse than expected

**Common Causes**:
1. **Incorrect collate function**: Using bimodal instead of trimodal
2. **Missing text modality**: One of dual texts not provided
3. **Config mismatch**: Model expects different input structure

**Solution**:
- Verify model_class in config matches actual model
- Check primary_text_name and secondary_text_name in config
- Ensure both text fields present in data
- Review model architecture requirements

### Issue 5: TensorBoard Plots Not Generated

**Symptom**: Missing ROC/PR curve files

**Common Causes**:
1. **Single class in data**: Can't plot with only one class
2. **Plotting library error**: Matplotlib/TensorBoard issue
3. **Insufficient data**: Too few samples for meaningful curves

**Solution**:
- Verify evaluation data has both classes (binary) or all classes (multiclass)
- Check logs for plotting errors
- Ensure at least 100 samples for reliable curves
- Install required plotting libraries

## References

### Related Scripts

- **Training Scripts:**
  - [`pytorch_training.py`](pytorch_training_script.md): PyTorch model training with Lightning
  - [`xgboost_training.py`](xgboost_training_script.md): XGBoost training for comparison
  - [`lightgbm_training.py`](lightgbm_training_script.md): LightGBM training

- **Evaluation Scripts:**
  - [`xgboost_model_eval.py`](xgboost_model_eval_script.md): XGBoost model evaluation
  - [`lightgbm_model_eval.py`](lightgbm_model_eval_script.md): LightGBM model evaluation
  - [`model_metrics_computation.py`](model_metrics_computation_script.md): Comprehensive metrics

- **Preprocessing Scripts:**
  - [`tabular_preprocessing.py`](tabular_preprocess_script.md): Tabular data preprocessing
  - [`temporal_sequence_normalization.py`](temporal_sequence_normalization_script.md): Sequence preprocessing

### Related Documentation

- **Contract**: `src/cursus/steps/contracts/pytorch_model_eval_contract.py` - Script contract specification
- **Step Builder**: [`slipbox/steps/pytorch_model_eval_step.md`](../steps/pytorch_model_eval_step.md) - Step builder implementation (if exists)
- **Config Class**: Model configuration in `projects/bsm_pytorch/docker/hyperparams/`
- **Step Specification**: PyTorch model evaluation step specification

### Related Design Documents

- **[PyTorch Model Evaluation Design](../1_design/pytorch_model_eval_design.md)**: Complete design for PyTorch model evaluation with GPU/CPU support, multi-modal architectures, and format preservation
- **[PyTorch Lightning Temporal Self Attention Design](../1_design/pytorch_lightning_temporal_self_attention_design.md)**: Deep learning architecture design for temporal models
- **[Temporal Self Attention Model Design](../1_design/temporal_self_attention_model_design.md)**: Model architecture for sequential data
- **[Data Format Preservation Patterns](../1_design/data_format_preservation_patterns.md)**: Format detection and preservation across pipeline scripts

### External References

- **[PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)**: Official PyTorch Lightning framework documentation
- **[PyTorch Documentation](https://pytorch.org/docs/stable/index.html)**: Core PyTorch library reference
- **[Hugging Face Transformers](https://huggingface.co/docs/transformers/index)**: BERT tokenizer and pre-trained models
- **[TensorBoard Documentation](https://www.tensorflow.org/tensorboard)**: Visualization tools for metrics and plots
- **[scikit-learn Metrics](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)**: Metrics computation reference

## Architecture Highlights

### Multi-Modal Architecture Support

**Supported Model Types**:
1. **Bimodal BERT**: Single text + tabular features
2. **Trimodal BERT**: Dual text + tabular features
3. **Trimodal Cross-Attention**: Dual text with cross-attention mechanism
4. **Trimodal Gate Fusion**: Dual text with gated fusion

**Key Features**:
- Automatic model architecture detection
- Dynamic collate function selection
- Separate handling of dual text modalities
- Flexible configuration via model artifacts

### PyTorch Lightning Integration

**Benefits**:
- Standardized inference utilities
- Consistent metrics computation
- Built-in visualization support
- GPU/CPU abstraction layer

**Workflow**:
```
1. Load Lightning model from checkpoint
2. Create dataset and dataloader
3. Use Lightning's model_inference utility
4. Compute metrics with Lightning utilities
5. Generate plots with Lightning plot functions
```

### Format Preservation Pattern

**Three-Function Pattern**:
1. `_detect_file_format()`: Detect format from extension
2. `load_dataframe_with_format()`: Load with format tracking
3. `save_dataframe_with_format()`: Save in same format

**Benefits**:
- Consistent I/O across pipeline
- No format conversion overhead
- Preserves data fidelity
- Storage optimization (Parquet)

### Preprocessing Pipeline Composition

**Operator Overloading Pattern**:
```python
pipeline = Processor1() >> Processor2() >> Processor3()
```

**Benefits**:
- Declarative pipeline definition
- Easy to modify and extend
- Type-safe composition
- Matches training preprocessing exactly

---
tags:
  - code
  - processing_script
  - model_inference
  - pytorch
  - deep_learning
keywords:
  - pytorch inference
  - model inference
  - batch prediction
  - multi-modal
  - text processing
  - format preservation
  - pure inference
topics:
  - model inference
  - pytorch predictions
  - modular ML pipelines
  - multi-GPU inference
language: python
date of note: 2025-11-22
---

# PyTorch Model Batch Inference Script Documentation

## Overview

The `pytorch_model_inference.py` script generates predictions from trained PyTorch Lightning models without computing evaluation metrics, enabling modular pipeline architectures where inference results can be cached, reused, and processed by different downstream components.

The script loads trained model artifacts (model weights, tokenizer, preprocessing parameters, feature configurations), applies the same preprocessing transformations used during training (text processing, risk table mapping, numerical imputation), and generates class probability predictions. It preserves the original input data structure while adding probability columns, supporting multiple output formats (CSV, TSV, Parquet) for flexible integration with downstream components.

Key capabilities:
- **Pure Inference**: Generates predictions without metrics computation for modular design
- **Multi-Modal Support**: Handles text, tabular, bimodal, and trimodal architectures
- **Model Artifact Loading**: Automatic extraction from model.tar.gz or direct loading
- **Text Processing Pipeline**: Full preprocessing with tokenization, normalization, and chunking
- **Embedded Preprocessing**: Self-contained risk table mapping and imputation processors
- **Format Preservation**: Maintains input format or converts to specified output format
- **Multi-GPU Support**: Distributed inference with synchronization for large-scale batches
- **Device Flexibility**: Automatic GPU detection or explicit CPU/GPU configuration
- **Binary and Multiclass**: Handles both classification scenarios with consistent output

## Purpose and Major Tasks

### Primary Purpose
Generate class probability predictions from trained PyTorch Lightning models by loading model artifacts, applying comprehensive preprocessing (text + tabular), and producing structured prediction outputs that preserve original data for downstream processing.

### Major Tasks

1. **Model Artifact Loading**: Load and extract trained model, tokenizer, preprocessing parameters, and feature configurations

2. **Data Loading**: Load inference data with automatic format detection (CSV, TSV, Parquet)

3. **Text Preprocessing Pipeline**: Build and apply text processing pipelines (HTML normalization, emoji removal, dialogue chunking, BERT tokenization)

4. **Risk Table Mapping**: Apply categorical feature transformations using trained risk tables

5. **Numerical Imputation**: Impute missing numerical values using trained imputation parameters

6. **Dataset Creation**: Create BSMDataset with complete preprocessing pipeline

7. **DataLoader Setup**: Configure DataLoader with appropriate collate functions for batch inference

8. **Device Configuration**: Setup single-GPU, multi-GPU, or CPU inference environment

9. **Prediction Generation**: Generate class probability predictions using PyTorch Lightning

10. **Multi-GPU Synchronization**: Coordinate distributed inference across multiple GPUs

11. **Output Formatting**: Structure predictions with original data and probability columns

12. **Multi-Format Saving**: Save predictions in specified format (CSV, TSV, Parquet)

13. **Health Checking**: Create success and health markers for pipeline monitoring

## Script Contract

### Entry Point
```
pytorch_model_inference.py
```

### Input Paths

| Path | Location | Description |
|------|----------|-------------|
| `model_input` | `/opt/ml/processing/input/model` | Trained PyTorch model artifacts |
| `inference_data` | `/opt/ml/processing/input/inference_data` | Inference data (NO label required) |

**Model Input Structure**:
```
/opt/ml/processing/input/model/
├── model.tar.gz (or extracted files below)
├── model.pth
├── model_artifacts.pth
├── risk_table_map.pkl
├── impute_dict.pkl
└── hyperparameters.json
```

### Output Paths

| Path | Location | Description |
|------|----------|-------------|
| `predictions_output` | `/opt/ml/processing/output/predictions` | Predictions with probabilities |

**Output Structure**:
```
/opt/ml/processing/output/predictions/
├── inference_predictions.{format}  # csv, tsv, or parquet
├── _SUCCESS                        # Success marker
└── _HEALTH                         # Health check file
```

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ID_FIELD` | Name of ID column in inference data | `"customer_id"` |

### Optional Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `"auto"` | Device configuration: `auto`, `cpu`, `gpu`, `cuda`, int (GPU count), or `[0,1,2,3]` (GPU IDs) |
| `ACCELERATOR` | `"auto"` | Accelerator type: `auto`, `cpu`, or `gpu` |
| `BATCH_SIZE` | `"32"` | Batch size for inference (overridden by model config if present) |
| `NUM_WORKERS` | `"0"` | Number of DataLoader workers |

**Device Configuration Examples**:
- `DEVICE="auto"` - Use all available GPUs or CPU
- `DEVICE="cpu"` - Force CPU usage
- `DEVICE="gpu"` or `DEVICE="cuda"` - Use single GPU (GPU 0)
- `DEVICE="4"` - Use 4 GPUs (0-3)
- `DEVICE="[0,2,3]"` - Use specific GPU IDs

### Job Arguments

| Argument | Type | Required | Description | Choices |
|----------|------|----------|-------------|---------|
| `--job_type` | `str` | Yes | Type of inference job | `inference`, `validation`, `testing` |

### Framework Dependencies

- **torch** >= 2.0.0 (PyTorch core)
- **lightning** >= 2.0.0 (PyTorch Lightning)
- **transformers** >= 4.30.0 (BERT tokenizer)
- **pandas** >= 1.3.0 (DataFrame operations)
- **numpy** >= 1.21.0 (Array operations)

## Input Data Structure

### Expected Input Format

**Inference Data**:
```
/opt/ml/processing/input/inference_data/
├── data.csv (or .tsv, .parquet)
└── _SUCCESS (optional marker)
```

### Required Columns

The script requires an ID column (specified by `ID_FIELD`) plus features used during training. **NO LABEL COLUMN REQUIRED** for pure inference.

**Typical Structure**:
- **ID Column**: Unique identifier for each record (configurable via `ID_FIELD`)
- **Text Columns**: Raw text fields (e.g., chat messages, events, descriptions)
- **Tabular Columns**: Numerical features for the tabular subnetwork
- **Categorical Columns**: Categorical features to be risk-table encoded

### Model Artifacts

**model.pth**: Trained PyTorch Lightning model checkpoint

**model_artifacts.pth**: Dictionary containing:
```python
{
    'config': {
        'model_class': 'bimodal_bert',  # or 'trimodal_bert', etc.
        'text_name': 'chat_messages',    # for bimodal
        'primary_text_name': 'chat',     # for trimodal
        'secondary_text_name': 'events', # for trimodal
        'tab_field_list': ['feature1', 'feature2', ...],
        'cat_field_list': ['cat1', 'cat2', ...],
        'max_sen_len': 512,
        'batch_size': 32,
        'tokenizer': 'bert-base-multilingual-cased',
        ...
    },
    'embedding_mat': np.ndarray,  # Optional embedding matrix
    'vocab': dict,                # Optional vocabulary
}
```

**risk_table_map.pkl**: Dictionary mapping categorical feature names to risk tables
```python
{
    'feature_name': {
        'bins': {'category1': 0.75, 'category2': 0.45, ...},
        'default_bin': 0.5
    }
}
```

**impute_dict.pkl**: Dictionary mapping feature names to imputation values
```python
{
    'numerical_feature1': 0.0,
    'numerical_feature2': 42.5,
    ...
}
```

**hyperparameters.json**: Model hyperparameters and training metadata
```json
{
    "model_class": "bimodal_bert",
    "is_binary": true,
    "max_sen_len": 512,
    ...
}
```

### Supported Input Formats

- **CSV**: Comma-separated values (`.csv`)
- **TSV**: Tab-separated values (`.tsv`)
- **Parquet**: Apache Parquet binary format (`.parquet`)

## Output Data Structure

### Output File Structure

**inference_predictions.{format}**:
- ID column only (NO label column)
- Added probability columns: `prob_class_0`, `prob_class_1`, ... `prob_class_N`
- Format determined by input format (CSV/TSV/Parquet)

### Output Columns

**Original Columns** (preserved):
- ID column (specified by `ID_FIELD`)
- NO label column (not required for pure inference)

**Added Prediction Columns**:
- `prob_class_0`: Probability for class 0
- `prob_class_1`: Probability for class 1
- `prob_class_N`: Probability for class N (multiclass)

### Output Format Examples

**CSV Output**:
```csv
customer_id,prob_class_0,prob_class_1
12345,0.85,0.15
67890,0.12,0.88
```

**Parquet Output**: Binary columnar format with same structure

### Success Markers

**_SUCCESS**: Empty file indicating successful completion

**_HEALTH**: Health check file with timestamp
```
healthy: 2025-11-22T10:45:00.123456
```

## Key Functions and Tasks

### Model Loading Component

#### `load_model_artifacts(model_dir) -> Tuple`

**Purpose**: Load trained PyTorch model and all preprocessing artifacts from directory

**Algorithm**:
```python
1. Check for model.tar.gz
   IF exists:
      Extract tar archive to model_dir
2. Load hyperparameters.json
3. Load model_artifacts.pth (config, embeddings, vocab)
4. Reconstruct tokenizer from config
5. Load trained model from model.pth
6. Set model to evaluation mode
7. Load preprocessing artifacts:
   a. Load risk_table_map.pkl
   b. Create risk processors
   c. Load impute_dict.pkl
   d. Create numerical processors
8. Return all artifacts as tuple
```

**Returns**: `(model, config, tokenizer, processors_dict)`

**Model Types Supported**:
- `bimodal_bert`: Text + Tabular with BERT
- `trimodal_bert`: Primary Text + Secondary Text + Tabular with BERT
- `bimodal_cnn`: Text + Tabular with CNN
- `trimodal_gate_fusion`: Text fusion with gating mechanism
- `bimodal_cross_attn`: Text + Tabular with cross-attention
- `bimodal_moe`: Mixture of Experts architecture

### Format Detection Component

#### `_detect_file_format(file_path) -> str`

**Purpose**: Detect data file format from extension

**Algorithm**: Same as XGBoost version (CSV, TSV, Parquet detection)

#### `load_dataframe_with_format(file_path) -> Tuple[pd.DataFrame, str]`

**Purpose**: Load DataFrame with automatic format detection

**Returns**: `(DataFrame, format_string)`

#### `save_dataframe_with_format(df, output_path, format_str) -> Path`

**Purpose**: Save DataFrame in specified format

**Supports**: CSV, TSV, Parquet (NO JSON for PyTorch inference)

### Text Preprocessing Component

#### `data_preprocess_pipeline(config, tokenizer) -> Tuple[AutoTokenizer, Dict]`

**Purpose**: Build text preprocessing pipelines based on model configuration

**Algorithm**:
```python
IF config has 'primary_text_name':  # TRIMODAL
   # Build primary text pipeline (full cleaning)
   primary_steps = config.get('primary_text_processing_steps', [
      'dialogue_splitter',
      'html_normalizer',
      'emoji_remover',
      'text_normalizer',
      'dialogue_chunker',
      'tokenizer'
   ])
   pipelines[primary_name] = build_text_pipeline_from_steps(...)
   
   # Build secondary text pipeline (minimal cleaning)
   secondary_steps = config.get('secondary_text_processing_steps', [
      'dialogue_splitter',
      'text_normalizer',
      'dialogue_chunker',
      'tokenizer'
   ])
   pipelines[secondary_name] = build_text_pipeline_from_steps(...)
   
ELSE:  # BIMODAL
   # Build single text pipeline
   steps = config.get('text_processing_steps', default_steps)
   pipelines[text_name] = build_text_pipeline_from_steps(...)

RETURN tokenizer, pipelines
```

**Returns**: `(tokenizer, pipelines_dict)` where pipelines_dict maps field names to processing pipelines

**Pipeline Components**:
- `dialogue_splitter`: Split text into dialogue turns
- `html_normalizer`: Remove HTML tags and normalize entities
- `emoji_remover`: Remove emoji characters
- `text_normalizer`: Normalize whitespace, punctuation
- `dialogue_chunker`: Chunk into manageable token sequences
- `tokenizer`: BERT tokenization with attention masks

#### `create_bsm_dataset(config, inference_data_dir, filename) -> BSMDataset`

**Purpose**: Create and initialize BSMDataset with missing value handling

**Algorithm**:
```python
1. Create BSMDataset from config and data file
2. Call fill_missing_value():
   - label_name = None  # NO label for inference
   - column_cat_name = config['cat_field_list']
3. Return initialized dataset
```

**Key Difference from Evaluation**: NO label column required or used

#### `apply_preprocessing_artifacts(bsm_dataset, processors, config) -> None`

**Purpose**: Apply numerical imputation and risk table mapping to dataset

**Algorithm**:
```python
1. Validate field types (numerical, categorical)
2. Apply numerical imputation processors:
   FOR each feature in numerical_processors:
      IF feature in dataset:
         bsm_dataset.add_pipeline(feature, processor)
3. Identify text fields to exclude from risk mapping
4. Apply risk table mapping processors (excluding text):
   FOR each feature in risk_processors:
      IF feature NOT in text_fields:
         IF feature in dataset:
            bsm_dataset.add_pipeline(feature, processor)
```

**Critical**: Text fields excluded from risk mapping to prevent overwriting tokenized text

### DataLoader Creation Component

#### `create_dataloader(bsm_dataset, config) -> DataLoader`

**Purpose**: Create DataLoader with appropriate collate function

**Algorithm**:
```python
1. Build collate function with keys from config:
   - input_ids_key = config.get('text_input_ids_key', 'input_ids')
   - attention_mask_key = config.get('text_attention_mask_key', 'attention_mask')
2. Get batch_size from config (default 32)
3. Create DataLoader:
   - collate_fn = bsm_collate_batch
   - batch_size = from config
   - shuffle = False (inference order matters)
4. Return configured DataLoader
```

**Collate Function**: Handles batching of:
- Tokenized text (input_ids, attention_mask)
- Tabular features (numerical arrays)
- Categorical features (already encoded)

### Device Configuration Component

#### `setup_device_environment(device) -> Tuple[Union[str, int, List[int]], str]`

**Purpose**: Setup device environment for inference with GPU/CPU detection

**Algorithm**:
```python
IF device == "auto":
   IF torch.cuda.is_available():
      gpu_count = torch.cuda.device_count()
      device_setting = gpu_count  # Use all GPUs
      accelerator = "gpu"
   ELSE:
      device_setting = "cpu"
      accelerator = "cpu"
ELIF device == "cpu":
   device_setting = "cpu"
   accelerator = "cpu"
ELIF device in ["cuda", "gpu"]:
   device_setting = 1  # Single GPU
   accelerator = "gpu"
ELIF isinstance(device, int):
   device_setting = device  # Use N GPUs
   accelerator = "gpu"
ELIF isinstance(device, list):
   device_setting = device  # Use specific GPU IDs
   accelerator = "gpu"

# Enable optimizations if using GPU
IF accelerator == "gpu":
   torch.backends.cudnn.benchmark = True

RETURN device_setting, accelerator
```

**Returns**: `(device_setting, accelerator_string)`

**GPU Configuration**:
- Logs GPU names and memory info
- Enables CuDNN benchmarking for optimization
- Supports single-GPU, multi-GPU, and CPU modes

### Prediction Generation Component

#### `generate_predictions(model, dataloader, device, accelerator) -> np.ndarray`

**Purpose**: Generate class probability predictions using PyTorch Lightning

**Algorithm**:
```python
1. Determine if multi-GPU inference:
   is_multi_gpu = (isinstance(device, int) AND device > 1) OR
                  (isinstance(device, list) AND len(device) > 1)
2. Call Lightning's model_inference utility:
   y_pred, _ = model_inference(
      model,
      dataloader,
      accelerator=accelerator,
      device=device,
      model_log_path=None  # No logging during inference
   )
3. Return prediction array
```

**Returns**: `np.ndarray` of shape `(n_samples, n_classes)`

**Multi-GPU Handling**:
- Distributed Data Parallel (DDP) if multiple GPUs
- Automatic synchronization across ranks
- Main process collects all predictions

### Output Saving Component with Multi-GPU Synchronization

#### `save_predictions(ids, y_prob, id_col, output_dir, input_format) -> None`

**Purpose**: Save predictions with ID column in specified format

**Algorithm**:
```python
1. Create DataFrame with ID column
2. Add probability columns:
   FOR i in range(n_classes):
      out_df[f"prob_class_{i}"] = y_prob[:, i]
3. Save using save_dataframe_with_format()
4. Log saved file path
```

**Critical Difference from XGBoost**: NO label column in output (pure inference)

#### `run_batch_inference(...) -> None`

**Purpose**: Orchestrate complete inference pipeline with multi-GPU support

**Algorithm**:
```python
1. Store IDs before preprocessing
2. Preprocess data and create DataLoader
3. Setup device environment
4. Generate predictions (ALL ranks participate in DDP)
5. IF is_main_process():
      Save predictions  # Only main process writes
   ELSE:
      Skip post-processing
6. IF dist.is_initialized():
      dist.barrier()  # Synchronize all ranks
7. Complete
```

**Multi-GPU Synchronization**:
- All ranks participate in inference
- Only main process (rank 0) performs I/O operations
- Barrier ensures all ranks wait for main process completion
- Prevents race conditions when writing to same files

## Algorithms and Data Structures

### Algorithm 1: Multi-Modal Text Processing Pipeline

**Problem**: Different text fields require different preprocessing levels (e.g., chat messages need full cleaning, events need minimal)

**Solution Strategy**: Configurable pipeline builder with field-specific steps

**Algorithm**:
```python
FOR each text field in config:
   1. Get processing steps for field:
      IF trimodal AND field == primary:
         steps = primary_text_processing_steps
      ELIF trimodal AND field == secondary:
         steps = secondary_text_processing_steps
      ELSE:
         steps = text_processing_steps
   
   2. Build pipeline from steps:
      pipeline = []
      FOR step_name in steps:
         IF step_name == 'dialogue_splitter':
            pipeline.append(DialogueSplitterProcessor())
         ELIF step_name == 'html_normalizer':
            pipeline.append(HTMLNormalizerProcessor())
         ELIF step_name == 'emoji_remover':
            pipeline.append(EmojiRemoverProcessor())
         ELIF step_name == 'text_normalizer':
            pipeline.append(TextNormalizationProcessor())
         ELIF step_name == 'dialogue_chunker':
            pipeline.append(DialogueChunkerProcessor(
               tokenizer=tokenizer,
               max_sen_len=max_sen_len,
               chunk_trancate=chunk_trancate,
               max_total_chunks=max_total_chunks
            ))
         ELIF step_name == 'tokenizer':
            pipeline.append(BertTokenizeProcessor(
               tokenizer=tokenizer,
               max_sen_len=max_sen_len,
               input_ids_key=input_ids_key,
               attention_mask_key=attention_mask_key
            ))
   
   3. Register pipeline for field:
      pipelines[field_name] = pipeline

RETURN pipelines
```

**Flexibility**: Each text field can have custom preprocessing pipeline

**Use Cases**:
- Chat messages: Full cleaning (HTML, emoji, normalization)
- Event logs: Minimal cleaning (just normalization)
- Product descriptions: Medium cleaning (HTML but keep special chars)

### Algorithm 2: Multi-GPU Distributed Inference with Synchronization

**Problem**: Coordinate inference across multiple GPUs while ensuring only one process writes output

**Solution Strategy**: DDP with main process I/O and barrier synchronization

**Algorithm**:
```python
# Phase 1: All ranks participate in inference
y_pred, _ = model_inference(
   model,
   dataloader,
   accelerator="gpu",
   device=4  # 4 GPUs: ranks 0, 1, 2, 3
)
# Each rank processes 1/4 of data
# Lightning automatically gathers predictions to rank 0

# Phase 2: Only main process (rank 0) does I/O
IF is_main_process():  # rank == 0
   # Write predictions to disk
   save_predictions(ids, y_pred, id_col, output_dir, format)
   # Create success markers
   create_health_check_file(...)
ELSE:  # ranks 1, 2, 3
   # Skip I/O operations
   pass

# Phase 3: Synchronize all ranks
IF dist.is_initialized():
   dist.barrier()  # All ranks wait here
   # Ensures main process completes I/O before script exits

# All ranks can now exit safely
```

**Benefits**:
- Parallel inference (4x speedup with 4 GPUs)
- No race conditions (only rank 0 writes)
- Safe script termination (barrier waits for I/O)

**Complexity**: 
- Inference: O(n/k × d×t) where k = number of GPUs
- I/O: O(n×f) (only rank 0)
- Synchronization: O(k) (barrier complexity)

### Algorithm 3: BSMDataset Pipeline Registration

**Problem**: Apply multiple preprocessing operations to different columns in correct order

**Solution Strategy**: Pipeline pattern with ordered processor registration

**Algorithm**:
```python
# Phase 1: Text processing pipelines (first)
FOR text_field, pipeline in text_pipelines.items():
   bsm_dataset.add_pipeline(text_field, pipeline)
# Text fields now tokenized → input_ids + attention_mask

# Phase 2: Numerical imputation (second)
FOR feature, processor in numerical_processors.items():
   IF feature in dataset:
      bsm_dataset.add_pipeline(feature, processor)
# Numerical NaNs now imputed

# Phase 3: Risk table mapping (third, excluding text)
text_field_names = {text_name, primary_text_name, secondary_text_name}
FOR feature, processor in risk_processors.items():
   IF feature NOT in text_field_names:  # CRITICAL: Skip text
      IF feature in dataset:
         bsm_dataset.add_pipeline(feature, processor)
# Categorical features now risk-encoded

# Dataset ready for inference
```

**Order Matters**:
1. Text first (tokenization creates new columns)
2. Numerical second (simple value replacement)
3. Categorical last (must not overwrite tokenized text)

**Critical**: Text fields excluded from risk mapping prevents:
```python
# BAD: Risk table overwrites tokenized text
tokenized_text = [101, 2054, 2003, ...]  # BERT tokens
risk_mapped = 0.75  # Overwrites tokens!

# GOOD: Text fields excluded
tokenized_text = [101, 2054, 2003, ...]  # Preserved
categorical_feature = 0.75  # Only non-text features mapped
```

## Performance Characteristics

### Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|----------------|------------------|-------|
| Model Loading | O(m) | O(m) | m = model size (~100-500MB) |
| Tar Extraction | O(t) | O(t) | t = tar size (if needed) |
| Data Loading | O(n×f) | O(n×f) | n = rows, f = features |
| Text Tokenization | O(n×l×t) | O(n×s) | l = text length, t = token ops, s = max sequence |
| Risk Table Mapping | O(n×c) | O(n×f) | c = categorical features |
| Numerical Imputation | O(n×f) | O(n×f) | f = features |
| PyTorch Forward Pass | O(n×d×h) | O(b×s×h) | d = depth, h = hidden size, b = batch size |
| Multi-GPU Inference | O(n/k × d×h) | O(b×s×h) | k = number of GPUs |
| Output Saving | O(n×f) | O(n×f) | Writing to disk |
| **Total (Single GPU)** | **O(n×(l×t + d×h))** | **O(n×f + m)** | Dominated by model inference |
| **Total (Multi-GPU)** | **O(n/k × (l×t + d×h))** | **O(n×f + m)** | k-fold speedup |

### Memory Requirements

**Peak Memory Usage (Single GPU)**:
```
Total ≈ Model + Input Data + Batch + Predictions + Output Buffer

Example (10K records, 2 text fields, 50 tabular features, binary):
- Model: ~150-400 MB (BERT-based models)
- Input DataFrame: ~5 MB (10K × 50 × 8 bytes)
- Text batch: ~10-20 MB (batch_size × max_seq_len × hidden_size)
- Tabular batch: ~2 MB (batch_size × 50 × 8 bytes)
- Predictions array: ~0.16 MB (10K × 2 × 8 bytes)
- Output buffer: ~5.2 MB (input + predictions)
- GPU memory: ~2-4 GB (model + activations)
Total: ~180-430 MB RAM + 2-4 GB VRAM
```

**Multi-GPU Memory**: Each GPU holds full model but processes 1/k of data

### Processing Time Estimates

| Dataset Size | Text Fields | Model | GPUs | Time (typical) |
|--------------|-------------|-------|------|----------------|
| 1K records   | 1           | BimodalBERT | 1 | ~5-10 seconds |
| 10K records  | 1           | BimodalBERT | 1 | ~30-60 seconds |
| 100K records | 1           | BimodalBERT | 1 | ~5-10 minutes |
| 100K records | 1           | BimodalBERT | 4 | ~2-3 minutes |
| 1M records   | 1           | BimodalBERT | 8 | ~15-25 minutes |
| 1M records   | 2           | TrimodalBERT | 8 | ~25-40 minutes |

**Notes**:
- Times include data loading, preprocessing, inference, and output saving
- Text processing adds 30-50% overhead vs tabular-only models
- Multi-GPU provides near-linear speedup (k×) for large batches

## Error Handling

### Model Loading Errors

**Missing Model Artifacts**:
- **Cause**: Required files not found in model directory
- **Handling**: Raises `FileNotFoundError` with missing file list
- **User Action**: Verify model training completion and output path

**Tokenizer Loading Failure**:
- **Cause**: Invalid tokenizer name in config or network issues
- **Handling**: Raises `OSError` from transformers library
- **User Action**: Check tokenizer name and internet connectivity

### Data Loading Errors

**No Inference Data**:
- **Cause**: No CSV/TSV/Parquet files in inference_data directory
- **Handling**: Raises `RuntimeError` with error message
- **User Action**: Verify inference_data input path

**Missing Required Columns**:
- **Severity**: Critical
- **Handling**: Raises `KeyError` when accessing missing columns
- **User Action**: Ensure input data has all required feature columns

### Preprocessing Errors

**Text Processing Failures**:
- **Cause**: Invalid text format or encoding issues
- **Handling**: Logs warning, replaces invalid text with empty string
- **Impact**: May reduce prediction quality for affected records

**Risk Table Missing Category**:
- **Handling**: Uses `default_bin` value from risk table
- **Impact**: Graceful fallback for unseen categories

### GPU Errors

**CUDA Out of Memory**:
- **Cause**: Batch size too large for available GPU memory
- **Handling**: Raises `torch.cuda.OutOfMemoryError`
- **User Action**: Reduce batch size in config or use gradient checkpointing

**Multi-GPU Initialization Failure**:
- **Cause**: DDP setup issues or GPU unavailable
- **Handling**: Falls back to single GPU or CPU
- **User Action**: Check GPU availability and NCCL configuration

## Best Practices

### For Production Deployments

1. **Use Multi-GPU for Large Datasets**: Significant speedup
   ```bash
   export DEVICE="4"  # Use 4 GPUs
   ```

2. **Optimize Batch Size**: Balance memory and throughput
   ```python
   # In config or env var
   batch_size = 32  # Good for single GPU
   batch_size = 64  # Good for multi-GPU
   ```

3. **Use Parquet Format**: More efficient for large datasets
   ```bash
   # Input and output both Parquet
   ```

4. **Monitor GPU Memory**: Track usage to optimize batch size
   ```bash
   nvidia-smi -l 1  # Monitor GPU usage
   ```

### For Development and Testing

1. **Start with CPU**: Easier to debug
   ```bash
   export DEVICE="cpu"
   ```

2. **Use Small Samples**: Test with subset before full dataset
   ```bash
   head -n 1001 large_data.csv > test_data.csv
   ```

3. **Single GPU First**: Test before scaling to multi-GPU
   ```bash
   export DEVICE="1"  # Single GPU
   ```

### For Performance Optimization

1. **Use Mixed Precision**: Enable AMP for faster inference
   - Configure in model training
   - Automatic in Lightning

2. **Optimize Text Processing**: Reduce max_sen_len if possible
   ```python
   # In config
   max_sen_len = 256  # Instead of 512
   ```

3. **Batch Processing**: Split very large datasets
   ```bash
   split -l 100000 large_data.csv chunk_
   ```

## Example Configurations

### Example 1: Standard Bimodal Inference (Single GPU)
```bash
export ID_FIELD="customer_id"
export DEVICE="gpu"

python pytorch_model_inference.py --job_type inference
```

**Use Case**: Standard bimodal (text + tabular) inference on single GPU

**Expected Output**: `inference_predictions.csv` with customer_id, prob_class_0, prob_class_1

### Example 2: Trimodal Inference with Multi-GPU
```bash
export ID_FIELD="transaction_id"
export DEVICE="4"  # Use 4 GPUs

python pytorch_model_inference.py --job_type inference
```

**Use Case**: Large-scale trimodal (primary text + secondary text + tabular) inference

**Benefits**:
- 4x speedup with 4 GPUs
- Efficient for datasets > 100K records
- Automatic DDP coordination

**Expected Output**: `inference_predictions.parquet` (format preserved from input)

### Example 3: CPU Inference for Testing
```bash
export ID_FIELD="sample_id"
export DEVICE="cpu"

python pytorch_model_inference.py --job_type validation
```

**Use Case**: Testing and debugging without GPU

**Benefits**:
- Easier to debug
- No GPU memory constraints
- Reproducible results

### Example 4: Specific GPU Selection
```bash
export ID_FIELD="customer_id"
export DEVICE="[0,2,3]"  # Use GPUs 0, 2, 3 (skip GPU 1)

python pytorch_model_inference.py --job_type inference
```

**Use Case**: Multi-GPU inference while leaving one GPU for other tasks

**Benefits**:
- Flexible GPU allocation
- Avoids interfering with other processes
- Optimal resource utilization

## Integration Patterns

### Upstream Integration

```
PyTorchTraining
   ↓ (outputs: model artifacts in model.tar.gz)
PyTorchModelInference
   ↓ (outputs: predictions with probabilities)
```

**Training Output → Inference Input**:
- Model artifacts: `model.pth`, `model_artifacts.pth`, `risk_table_map.pkl`, `impute_dict.pkl`, `hyperparameters.json`
- Packaged in `model.tar.gz` or extracted
- Tokenizer reconstructed from config

### Downstream Integration

```
PyTorchModelInference
   ↓ (outputs: inference_predictions.csv with ID + prob_class_*)
ModelMetricsComputation
   ↓ (outputs: comprehensive metrics)
```

**OR**

```
PyTorchModelInference
   ↓ (outputs: inference_predictions.csv with prob_class_*)
ModelCalibration/PercentileModelCalibration
   ↓ (outputs: calibrated model or percentile mapping)
```

### Complete Pipeline Flow

```
TabularPreprocessing → PyTorchTraining → PyTorchModelInference →
[Branch 1] → ModelMetricsComputation → ModelWikiGenerator
[Branch 2] → ModelCalibration → Package → Registration
```

**Key Integration Points**:
1. **Training → Inference**: Model artifacts + preprocessing parameters
2. **Inference → Metrics**: Predictions (if labels available in separate file)
3. **Inference → Calibration**: Predictions (if labels available in separate file)
4. **Inference → Multiple Downstream**: Modular design allows caching and reuse

### Modular Design Benefits

**Cache and Reuse**:
- Generate predictions once
- Use same predictions for multiple downstream tasks
- Avoid redundant inference computation

**Parallel Processing**:
- Run metrics computation and calibration in parallel
- Both consume same inference output
- Reduces overall pipeline time

**Testing and Validation**:
- Test metrics computation without re-running inference
- Validate calibration with frozen predictions
- Reproducible results

## Troubleshooting

### Issue: CUDA Out of Memory

**Symptom**: 
```
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 136.00 MiB. 
GPU 0 has a total capacity of 15.77 GiB of which 58.50 MiB is free.
```

**Common Causes**:
1. Batch size too large for GPU memory
2. Text sequences too long (high `max_sen_len`)
3. Multiple text fields consuming memory
4. Model too large for GPU

**Solutions**:

1. **Reduce Batch Size**:
   ```python
   # In model config
   batch_size = 16  # Instead of 32
   batch_size = 8   # For very large models
   ```

2. **Reduce Text Sequence Length**:
   ```python
   # In model config
   max_sen_len = 256  # Instead of 512
   ```

3. **Use Gradient Checkpointing** (if available in model):
   - Trades compute for memory
   - Configured during training

4. **Use Multiple GPUs**:
   ```bash
   export DEVICE="2"  # Split across 2 GPUs
   ```

5. **Switch to CPU** (slower but no memory limit):
   ```bash
   export DEVICE="cpu"
   ```

6. **Process in Batches**:
   ```bash
   # Split data into smaller chunks
   split -l 10000 large_data.csv chunk_
   ```

### Issue: Token Sequence Length Warnings

**Symptom**: 
```
Token indices sequence length is longer than the specified maximum sequence 
length for this model (854 > 512). Running this sequence through the model 
will result in indexing errors
```

**Common Causes**:
1. Text too long for configured `max_sen_len`
2. Dialogue chunker not working properly
3. Unexpected text format

**Solutions**:

1. **Verify Text Preprocessing**:
   - Check if dialogue_chunker is in pipeline
   - Verify chunk_trancate setting

2. **Reduce max_sen_len**:
   ```python
   # In training config (must retrain)
   max_sen_len = 256
   ```

3. **Text is Truncated** (not an error):
   - Model will truncate automatically
   - May lose information from long texts
   - Consider pre-truncating or summarizing

### Issue: Missing Logger Folder Warnings

**Symptom**: 
```
Missing logger folder: /opt/ml/output/data/checkpoints/tensorboard_logs
```

**Cause**: Lightning looking for training-time log directories during inference

**Solution**: 
- **Ignore**: These warnings are harmless during inference
- No tensorboard logging needed for inference
- Script completes successfully despite warnings

### Issue: NCCL Initialization Warnings

**Symptom**: 
```
NCCL WARN NET/OFI Failed to initialize sendrecv protocol
NCCL WARN NET/OFI aws-ofi-nccl initialization failed
```

**Common Causes**:
1. EFA (Elastic Fabric Adapter) not available or configured
2. Using instance type without EFA support
3. NCCL falling back to alternative protocol

**Solution**:
- **Usually harmless**: NCCL will fall back to alternative communication
- Performance may be slightly reduced but still functional
- For optimal multi-GPU performance, use EFA-enabled instances (p4d, p5)

### Issue: Model Artifacts Not Found

**Symptom**: `FileNotFoundError: Model artifacts not found`

**Common Causes**:
1. Training didn't complete successfully
2. Model path misconfigured
3. model.tar.gz not properly packaged

**Solution**:
1. Check training logs for completion
2. Verify model output path matches inference input:
   ```bash
   ls -la /opt/ml/processing/input/model/
   ```
3. If tar.gz exists, verify contents:
   ```bash
   tar -tzf /opt/ml/processing/input/model/model.tar.gz
   ```

### Issue: Text Field Not Tokenized

**Symptom**: Raw text strings in output instead of predictions

**Common Causes**:
1. Text field name mismatch between config and data
2. Text processing pipeline not applied
3. Preprocessing step failed silently

**Solution**:
1. Verify field names match config:
   ```python
   # Check config
   text_name = config.get('text_name')  # or primary_text_name
   # Check data
   print(df.columns.tolist())
   ```

2. Check preprocessing logs for errors

3. Verify text processing steps in config

### Issue: All Predictions Same Value

**Symptom**: All predictions show same probability (e.g., all 0.5)

**Common Causes**:
1. Model not properly loaded
2. Features not properly preprocessed
3. All features missing or invalid

**Solution**:
1. Verify model loading:
   ```python
   print(model)  # Should show model architecture
   ```

2. Check preprocessing:
   ```python
   # After preprocessing
   print(df[tab_field_list].describe())
   print(df[tab_field_list].isna().sum())
   ```

3. Verify text tokenization:
   - Text should be converted to input_ids
   - Check first few samples manually

## References

### Related Scripts

- [`pytorch_training.py`](pytorch_training_script.md): Training script that produces model artifacts consumed by this script
- [`pytorch_model_eval.py`](pytorch_model_eval_script.md): Evaluation script that computes metrics from predictions (with labels)
- [`pytorch_inference_handler.py`](pytorch_inference_handler_script.md): Real-time endpoint handler for SageMaker inference
- [`xgboost_model_inference.py`](xgboost_model_inference_script.md): Similar inference script for XGBoost models
- [`lightgbm_model_inference.py`](lightgbm_model_inference_script.md): Similar inference script for LightGBM models

### Related Documentation

- **Model Architectures**: See `slipbox/models/lightning_models/` for architecture documentation
  - [`pl_bimodal_bert.md`](../models/lightning_models/pl_bimodal_bert.md)
  - [`pl_trimodal_bert.md`](../models/lightning_models/pl_trimodal_bert.md)
  - [`pl_bimodal_gate_fusion.md`](../models/lightning_models/pl_bimodal_gate_fusion.md)
- **Processing Components**: See `slipbox/scripts/` for processor documentation
  - Text processing: DialogueSplitter, HTMLNormalizer, EmojiRemover, etc.
  - Risk table mapping: RiskTableMappingProcessor
  - Numerical imputation: NumericalVariableImputationProcessor

### Related Design Documents

- **[PyTorch Inference Design](../1_design/pytorch_inference_design.md)**: Comprehensive design document covering pure inference architecture, multi-modal support, and multi-GPU patterns
- **[Multi-Modal Architecture Design](../1_design/multimodal_architecture_design.md)**: Design patterns for bimodal and trimodal neural architectures
- **[Text Processing Pipeline Design](../1_design/text_processing_pipeline_design.md)**: Text preprocessing component design and implementation patterns

### External References

- [PyTorch Lightning Documentation](https://lightning.ai/docs/pytorch/stable/): Official PyTorch Lightning documentation
- [Transformers Documentation](https://huggingface.co/docs/transformers/): Hugging Face Transformers library for BERT
- [PyTorch Distributed](https://pytorch.org/docs/stable/distributed.html): PyTorch distributed training and inference
- [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/): NVIDIA Collective Communications Library for multi-GPU
- [Apache Parquet](https://parquet.apache.org/docs/): Columnar storage format documentation

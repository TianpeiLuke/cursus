---
tags:
  - code
  - scripts
  - tokenizer_training_script
  - script_documentation
  - natural_language_processing
keywords:
  - tokenizer training
  - BPE tokenizer
  - byte pair encoding
  - compression tokenizer
  - text processing
  - names3risk
topics:
  - pipeline scripts
  - tokenizer training
  - text preprocessing
language: python
date of note: 2026-01-13
---

# Tokenizer Training Script Documentation

## Overview

The `tokenizer_training.py` script trains a custom Byte Pair Encoding (BPE) tokenizer optimized for customer name data in fraud detection systems. It automatically tunes the vocabulary size to achieve a target compression ratio, making it specifically designed for the Names3Risk project where efficient text representation is critical.

The script provides a production-ready tokenizer training pipeline with support for multi-format data loading (CSV, TSV, Parquet), automatic subdirectory detection for multi-split datasets, compression-based vocabulary tuning, and comprehensive artifact generation for downstream inference. It handles the complete workflow from data loading and text extraction through tokenizer training and artifact saving.

Key capabilities include:
- **Multi-Format Data Loading**: Automatic detection and loading of CSV, TSV, and Parquet files
- **Multi-Split Aggregation**: Automatically discovers and concatenates data from all subdirectories (train/val/test)
- **Compression-Optimized Training**: Automatically tunes vocabulary size to achieve target compression ratio
- **Names3Risk Specialization**: Optimized for customer name data with special tokens for missing values and delimiters
- **Comprehensive Artifact Generation**: Saves tokenizer in multiple formats (HuggingFace JSON, vocabulary dict, metadata)
- **Flexible Configuration**: Configurable via environment variables for easy pipeline integration
- **Format Preservation**: Maintains compatibility with upstream preprocessing output formats

## Purpose and Major Tasks

### Primary Purpose
Train a Byte Pair Encoding (BPE) tokenizer on customer name data that achieves optimal text compression while maintaining semantic information, specifically designed for the Names3Risk fraud detection system.

### Major Tasks

1. **Data Discovery and Loading**: Automatically detect subdirectories and data files across multiple formats
2. **Multi-File Aggregation**: Concatenate texts from all discovered files (train/val/test splits)
3. **Text Extraction**: Extract text field from loaded data with missing value handling
4. **Tokenizer Initialization**: Set up BPE tokenizer with special tokens optimized for Names3Risk
5. **Compression-Based Training**: Train tokenizer while auto-tuning vocabulary size for target compression
6. **Vocabulary Optimization**: Iteratively adjust vocabulary size to meet compression requirements
7. **Artifact Generation**: Save tokenizer in multiple formats for different use cases
8. **Metadata Recording**: Record training parameters and vocabulary statistics

## Script Contract

### Entry Point
```
tokenizer_training.py
```

### Input Paths

| Path | Location | Description |
|------|----------|-------------|
| `input_data` | `/opt/ml/processing/input` | Root directory containing data subdirectories or files |
| `train` | `/opt/ml/processing/input/train` | Training data files (optional subdirectory) |
| `val` | `/opt/ml/processing/input/val` | Validation data files (optional subdirectory) |
| `test` | `/opt/ml/processing/input/test` | Test data files (optional subdirectory) |

**Input Discovery Logic**:
1. Check if input path has subdirectories
2. If subdirectories exist: Load ALL files from ALL subdirectories
3. If no subdirectories: Load files from root directory
4. Supported file formats: `.csv`, `.tsv`, `.parquet`

**Input Structure Example 1** (with subdirectories - from tabular preprocessing):
```
/opt/ml/processing/input/
├── train/
│   └── train_processed_data.csv
├── val/
│   └── val_processed_data.csv
└── test/
    └── test_processed_data.parquet
```

**Input Structure Example 2** (without subdirectories - direct files):
```
/opt/ml/processing/input/
├── data.csv
└── additional_data.parquet
```

### Output Paths

| Path | Location | Description |
|------|----------|-------------|
| `model_artifacts_output` | `/opt/ml/processing/output` | Tokenizer artifacts directory |

**Output Contents**:
- `tokenizer.json`: HuggingFace Tokenizers format (main artifact, used for inference)
- `vocab.json`: Token-to-ID mapping dictionary (legacy compatibility)
- `tokenizer_metadata.json`: Configuration metadata (vocab_size, special_tokens, etc.)

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `TEXT_FIELD` | Column name containing text data | `"text"` |

### Optional Environment Variables

| Variable | Default | Description | Range |
|----------|---------|-------------|-------|
| `TARGET_COMPRESSION` | `"2.5"` | Target compression ratio for tokenizer | `1.5` - `5.0` |
| `MIN_FREQUENCY` | `"25"` | Minimum frequency threshold for BPE merges | `1` - `100` |
| `MAX_VOCAB_SIZE` | `"50000"` | Maximum vocabulary size limit | `1000` - `100000` |
| `USE_SECURE_PYPI` | `"false"` | Use secure CodeArtifact PyPI for package installation | `"true"`, `"false"` |

### Job Arguments

| Argument | Type | Required | Description | Options |
|----------|------|----------|-------------|---------|
| `--job_type` | `str` | Yes | Processing mode identifier | `"training"`, `"validation"`, `"testing"`, `"calibration"` |

**Note**: `job_type` is required by the SageMaker processing contract but not actively used by the tokenizer training logic.

## Input Data Structure

### Expected Input Format

**Option 1: From Tabular Preprocessing** (with subdirectories):
```
/opt/ml/processing/input/
├── train/
│   └── train_processed_data.csv
├── val/
│   └── val_processed_data.csv
└── test/
    └── test_processed_data.csv
```

**Option 2: Direct Input** (without subdirectories):
```
/opt/ml/processing/input/
└── training_texts.parquet
```

### Required Columns in Data Files

**Essential Column**:
- Text column (name specified by `TEXT_FIELD` env var): Contains text data for tokenizer training

**Example Data**:
```csv
text
"john.smith@example.com|John Smith|123 Main St"
"jane.doe@example.com|Jane Doe|456 Oak Ave"
"bob.lee@example.com|[MISSING]|789 Pine Rd"
```

**Text Format Requirements**:
- Can contain any UTF-8 text
- Missing values automatically handled
- Special characters (e.g., `|`, `[MISSING]`) preserved and learned
- Typical for Names3Risk: concatenated fields with `|` separator

### Supported Data Formats

1. **CSV Files** (`.csv`): Comma-separated values
2. **TSV Files** (`.tsv`): Tab-separated values
3. **Parquet Files** (`.parquet`): Columnar format

**Format Detection**: Automatic based on file extension

### Multi-File Aggregation

The script automatically aggregates texts from multiple sources:

**Example**: 3 subdirectories with different formats
```python
Input:
  train/train_data.csv     → 50,000 texts
  val/val_data.tsv         → 10,000 texts
  test/test_data.parquet   → 10,000 texts

Output:
  Combined: 70,000 texts for tokenizer training
```

**Benefits**:
- Maximum data utilization across all splits
- More diverse vocabulary learning
- Better compression optimization
- Consistent tokenizer across all data

## Output Data Structure

### Tokenizer Artifacts Directory

```
/opt/ml/processing/output/
├── tokenizer.json              # Main HuggingFace tokenizer artifact
├── vocab.json                  # Token-to-ID mapping (legacy compatibility)
└── tokenizer_metadata.json     # Training metadata
```

### tokenizer.json

**Format**: HuggingFace Tokenizers JSON format

**Contents**:
- Model configuration (BPE parameters)
- Complete vocabulary with token IDs
- Special tokens configuration
- Normalizer settings (NFKC Unicode)
- Pre-tokenizer configuration (Whitespace)
- Trained BPE merges

**Usage**: Load directly with HuggingFace Tokenizers library for inference

**Example**:
```python
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")
encoded = tokenizer.encode("john.smith@example.com")
```

### vocab.json

**Format**: JSON dictionary

**Contents**: Token string to ID mapping
```json
{
  "[PAD]": 0,
  "[CLS]": 1,
  "[UNK]": 2,
  "[BOS]": 3,
  "[EOS]": 4,
  "[MISSING]": 5,
  "|": 6,
  "a": 7,
  "e": 8,
  "john": 123,
  "smith": 456,
  ...
}
```

**Usage**: Legacy compatibility, quick vocabulary inspection

### tokenizer_metadata.json

**Format**: JSON metadata

**Contents**: Training configuration and statistics
```json
{
  "vocab_size": 8543,
  "model_type": "BPE",
  "special_tokens": ["[CLS]", "[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MISSING]", "|"],
  "normalizer": "NFKC",
  "pre_tokenizer": "Whitespace",
  "pad_token": 0,
  "cls_token": 1,
  "min_frequency": 25
}
```

**Usage**: Model cards, documentation, reproducibility

## Key Functions and Tasks

### Data Loading Component

#### `load_train_texts(train_data_path, text_field, log)`
**Purpose**: Load and aggregate training texts from single file or multiple subdirectories

**Algorithm**:
```python
1. Determine if path is file or directory
2. If file:
   a. Load file directly
   b. Extract text column
   c. Return texts
3. If directory:
   a. Check for subdirectories
   b. If subdirectories exist:
      - Iterate through each subdirectory
      - Find all supported files (.csv, .tsv, .parquet)
      - Load each file and extract texts
      - Aggregate all texts
   c. If no subdirectories:
      - Find all supported files in root
      - Load and aggregate
4. Handle missing values (drop NaN)
5. Log summary statistics
6. Return complete text list
```

**Parameters**:
- `train_data_path` (str): Path to data directory or file
- `text_field` (str): Name of text column (default: `"text"`)
- `log` (Callable): Logging function

**Returns**: `List[str]` - List of training texts

**Multi-Format Handling**:
```python
def load_file(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(file_path)
    elif suffix == ".csv":
        return pd.read_csv(file_path)
    elif suffix == ".tsv":
        return pd.read_csv(file_path, sep="\t")
```

**Complexity**: O(n * m) where n=number of files, m=average file size

**Example Log Output**:
```
Loading training data from /opt/ml/processing/input
Found 3 subdirectories: ['train', 'val', 'test']
Found file in train/: train_processed_data.csv
Found file in val/: val_processed_data.csv
Found file in test/: test_processed_data.parquet
Loading and concatenating 3 file(s)...
  Loaded 50,000 texts from train/train_processed_data.csv
  Loaded 10,000 texts from val/val_processed_data.csv
  Loaded 10,000 texts from test/test_processed_data.parquet
Total texts loaded: 70,000
Sample text: john.smith@example.com|John Smith|123 Main St...
```

### Tokenizer Training Component

#### `CompressionBPETokenizer.train(texts, target_compression, max_vocab_size)`
**Purpose**: Train BPE tokenizer with automatic vocabulary size tuning for target compression

**Algorithm**:
```python
1. Initialize BPE tokenizer with special tokens:
   - [CLS]: Classification token
   - [PAD]: Padding token
   - [UNK]: Unknown token
   - [BOS]: Beginning of sequence
   - [EOS]: End of sequence
   - [MISSING]: Missing value indicator (Names3Risk specific)
   - |: Field separator (Names3Risk specific)

2. Set normalizer: NFKC Unicode normalization

3. Set pre-tokenizer: Whitespace splitting

4. Train BPE model:
   a. Start with initial vocabulary size estimate
   b. Train BPE with min_frequency threshold
   c. Compute compression ratio on sample texts
   d. If compression < target:
      - Increase vocabulary size
      - Retrain
   e. If compression > target:
      - Decrease vocabulary size
      - Retrain
   f. Repeat until target compression achieved or max_vocab_size reached

5. Finalize tokenizer with optimal vocabulary

6. Return trained tokenizer
```

**Parameters**:
- `texts` (List[str]): Training texts
- `target_compression` (float): Target compression ratio (default: 2.5)
- `max_vocab_size` (int): Maximum vocabulary size (default: 50,000)

**Returns**: Trained `CompressionBPETokenizer` instance

**Compression Calculation**:
```python
compression_ratio = sum(len(text) for text in texts) / sum(len(tokens) for tokens in encoded_texts)
```

**Example**:
- Input text: `"john.smith@example.com"` (21 characters)
- Encoded tokens: `["john", ".", "smith", "@", "example", ".", "com"]` (7 tokens)
- Compression: 21 / 7 = 3.0

**Optimization Strategy**:
- Binary search for optimal vocabulary size
- Balances compression vs. vocabulary size
- Ensures semantic preservation

**Complexity**: O(n * m * log(v)) where:
- n = number of training texts
- m = average text length
- v = vocabulary size (due to iterative optimization)

### Artifact Saving Component

#### `save_tokenizer_artifacts(tokenizer, output_dir, log)`
**Purpose**: Save trained tokenizer in multiple formats for different use cases

**Algorithm**:
```python
1. Create output directory if not exists

2. Save HuggingFace tokenizer format:
   a. Call tokenizer.save(output_dir / "tokenizer.json")
   b. Includes full model configuration
   c. Ready for immediate use in inference

3. Extract and save vocabulary:
   a. Get vocabulary dict from tokenizer
   b. Save as vocab.json (token → ID mapping)
   c. Pretty-print for human readability

4. Create and save metadata:
   a. Collect vocab_size
   b. List special_tokens
   c. Record normalizer type
   d. Record pre_tokenizer type
   e. Save min_frequency parameter
   f. Save as tokenizer_metadata.json

5. Log all saved files
```

**Parameters**:
- `tokenizer` (CompressionBPETokenizer): Trained tokenizer
- `output_dir` (str): Output directory path
- `log` (Callable): Logging function

**Returns**: None (saves files to disk)

**File Sizes** (approximate):
- `tokenizer.json`: 500KB - 5MB (depends on vocab_size)
- `vocab.json`: 100KB - 1MB
- `tokenizer_metadata.json`: < 1KB

**Example Output**:
```
Saved tokenizer to /opt/ml/processing/output/tokenizer.json
Saved vocabulary to /opt/ml/processing/output/vocab.json
Saved metadata to /opt/ml/processing/output/tokenizer_metadata.json
```

### Main Orchestration Component

#### `main(input_paths, output_paths, environ_vars, job_args, logger_func)`
**Purpose**: Orchestrate complete tokenizer training workflow

**Workflow**:
```
1. Extract Configuration
   ↓
2. Load Training Texts (with multi-file aggregation)
   ↓
3. Validate Data (check for empty texts)
   ↓
4. Initialize CompressionBPETokenizer
   ↓
5. Train Tokenizer (with compression tuning)
   ↓
6. Save Artifacts (tokenizer, vocab, metadata)
   ↓
7. Log Results Summary
```

**Parameters**:
- `input_paths` (Dict[str, str]): Input path mapping
- `output_paths` (Dict[str, str]): Output path mapping
- `environ_vars` (Dict[str, str]): Environment variables
- `job_args` (argparse.Namespace): Command line arguments
- `logger_func` (Callable): Logging function

**Returns**: `Dict[str, any]` - Training results and metadata

**Result Dictionary**:
```python
{
    "vocab_size": 8543,
    "num_training_texts": 70000,
    "target_compression": 2.5,
    "min_frequency": 25,
    "pad_token": 0,
    "cls_token": 1
}
```

**Error Handling**:
- FileNotFoundError: No data files found
- ValueError: Empty text list or missing TEXT_FIELD
- Exception: General training failure with traceback

**Complexity**: O(n * m) dominated by tokenizer training

## Algorithms and Data Structures

### Byte Pair Encoding (BPE) Algorithm

**Purpose**: Learn subword vocabulary that balances compression and semantic preservation

**Algorithm**:
```python
1. Initialize vocabulary with individual characters

2. Compute all adjacent character pair frequencies in training corpus

3. Repeat until vocabulary size or frequency threshold reached:
   a. Find most frequent character pair (e.g., "th" appears 1000 times)
   b. Merge this pair into single token ("th")
   c. Replace all occurrences in corpus
   d. Update pair frequencies
   e. Add new token to vocabulary

4. Result: Vocabulary containing:
   - Individual characters (base coverage)
   - Common subwords (e.g., "ing", "tion")
   - Frequent words (e.g., "the", "and")
   - Domain-specific tokens (e.g., "smith", ".com")
```

**Example Progression**:
```
Iteration 0: Vocabulary = {a, b, c, d, e, ...}
Iteration 1: Found "th" (freq=1000) → Vocabulary += {"th"}
Iteration 2: Found "th" + "e" = "the" (freq=800) → Vocabulary += {"the"}
Iteration 3: Found "in" (freq=750) → Vocabulary += {"in"}
...
Final: Vocabulary = 8543 tokens
```

**Complexity**: O(n * m * v) where:
- n = number of merge iterations
- m = total corpus size
- v = vocabulary size at each iteration

**Memory**: O(v + c) where:
- v = vocabulary size
- c = corpus size (for frequency counting)

### Compression Ratio Optimization

**Purpose**: Automatically tune vocabulary size to achieve target compression ratio

**Algorithm**:
```python
def optimize_compression(texts, target_compression, min_vocab, max_vocab):
    """Binary search for optimal vocabulary size."""
    low, high = min_vocab, max_vocab
    best_vocab_size = max_vocab
    
    while low <= high:
        mid = (low + high) // 2
        
        # Train tokenizer with mid vocabulary size
        tokenizer = train_bpe(texts, vocab_size=mid)
        
        # Compute compression on sample
        compression = compute_compression(tokenizer, texts[:1000])
        
        if abs(compression - target_compression) < 0.1:
            # Close enough, return
            return mid
        elif compression < target_compression:
            # Need more compression (larger vocab)
            low = mid + 1
        else:
            # Too much compression (smaller vocab)
            high = mid - 1
            best_vocab_size = mid
    
    return best_vocab_size
```

**Complexity**: O(log(v) * n * m) where:
- log(v) = binary search iterations
- n = number of training texts
- m = average text length

**Typical Behavior**:
- Target compression 2.5 → vocab_size ≈ 8,000 - 12,000
- Target compression 3.0 → vocab_size ≈ 15,000 - 20,000
- Higher compression = larger vocabulary needed

### Special Token Handling

**Purpose**: Reserve special tokens for task-specific semantics

**Special Tokens**:
```python
SPECIAL_TOKENS = [
    "[CLS]",      # ID=1: Classification/start token
    "[PAD]",      # ID=0: Padding for batching
    "[UNK]",      # ID=2: Unknown tokens
    "[BOS]",      # ID=3: Beginning of sequence
    "[EOS]",      # ID=4: End of sequence
    "[MISSING]",  # ID=5: Missing value indicator (Names3Risk)
    "|"           # ID=6: Field separator (Names3Risk)
]
```

**Usage in Names3Risk**:
- `[MISSING]`: Represents missing fields in concatenated text
- `|`: Separates different fields (email, name, address)

**Example Text with Special Tokens**:
```
Input: "john.smith@example.com|[MISSING]|123 Main St"

Tokens: ["john", ".", "smith", "@", "example", ".", "com", "|", "[MISSING]", "|", "123", "Main", "St"]

IDs: [123, 456, 789, 234, 567, 456, 890, 6, 5, 6, 345, 678, 901]
```

**Reserved ID Space**: First 7 IDs (0-6) reserved for special tokens

### Data Structure: Token Vocabulary

**Purpose**: Efficient mapping between tokens and IDs

**Structure**:
```python
class TokenVocabulary:
    token_to_id: Dict[str, int]  # Token string → ID
    id_to_token: List[str]       # ID → Token string
    vocab_size: int              # Total vocabulary size
    special_tokens: Dict[str, int]  # Special token mappings
```

**Operations**:
- `encode(text)`: O(m) where m=text length
- `decode(ids)`: O(n) where n=number of IDs
- `lookup(token)`: O(1) hash table lookup
- `reverse_lookup(id)`: O(1) array index

**Memory**: O(v) where v=vocabulary size

## Performance Characteristics

### Training Performance

| Dataset Size | Files | Training Time | Memory Usage | Vocab Size |
|--------------|-------|---------------|--------------|------------|
| 10K texts | 1 file | ~30s | ~500MB | ~5,000 |
| 50K texts | 3 files | ~2min | ~1GB | ~8,000 |
| 100K texts | 3 files | ~5min | ~2GB | ~12,000 |
| 500K texts | Multiple | ~20min | ~5GB | ~20,000 |
| 1M+ texts | Multiple | ~45min | ~10GB | ~30,000 |

**Note**: Times measured on ml.m5.xlarge instance (4 vCPUs, 16GB RAM)

### Compression Characteristics

| Target Compression | Typical Vocab Size | Use Case |
|-------------------|-------------------|----------|
| 1.5 - 2.0 | 5,000 - 8,000 | Character-heavy text |
| 2.0 - 2.5 | 8,000 - 12,000 | Balanced (Names3Risk default) |
| 2.5 - 3.0 | 12,000 - 18,000 | Word-heavy text |
| 3.0+ | 18,000+ | Very dense text |

**Compression-Vocab Size Trade-off**:
- Higher compression → Larger vocabulary → More memory
- Lower compression → Smaller vocabulary → Less precise

### File Format Performance

| Format | Load Time (100K rows) | File Size | Pros | Cons |
|--------|----------------------|-----------|------|------|
| CSV | ~5s | ~50MB | Human-readable | Slow, large |
| TSV | ~5s | ~50MB | Tab-delimited | Slow, large |
| Parquet | ~1s | ~10MB | Fast, compressed | Binary format |

**Recommendation**: Use Parquet for production, CSV for debugging

### Multi-File Aggregation Performance

| Files | Total Rows | Aggregation Time | Memory Peak |
|-------|-----------|------------------|-------------|
| 1 file | 50K | Baseline | 500MB |
| 3 files | 150K | +2s | 1.2GB |
| 5 files | 250K | +5s | 2GB |
| 10 files | 500K | +12s | 4GB |

**Overhead**: ~1-2s per additional file for loading and concatenation

## Configuration

### Environment Variable Details

#### TEXT_FIELD
**Description**: Column name containing text data in input files

**Required**: Yes

**Type**: String

**Example Values**:
- `"text"` (default from tabular preprocessing)
- `"customer_name"`
- `"dialogue"`

**Effect**: Determines which column to extract for tokenizer training

#### TARGET_COMPRESSION
**Description**: Target compression ratio for vocabulary optimization

**Required**: No (default: `"2.5"`)

**Type**: Float (as string)

**Range**: `1.5` - `5.0`

**Typical Values**:
- `"2.0"`: Lower compression, smaller vocabulary
- `"2.5"`: Balanced (recommended for Names3Risk)
- `"3.0"`: Higher compression, larger vocabulary

**Effect**: Higher values require larger vocabulary to achieve compression

#### MIN_FREQUENCY
**Description**: Minimum frequency threshold for BPE merge operations

**Required**: No (default: `"25"`)

**Type**: Integer (as string)

**Range**: `1` - `100`

**Typical Values**:
- `"1"`: Include all tokens (large vocabulary)
- `"25"`: Balanced filtering (recommended)
- `"50"`: Aggressive filtering (small vocabulary)

**Effect**: Higher values create smaller but less precise vocabularies

#### MAX_VOCAB_SIZE
**Description**: Maximum vocabulary size limit

**Required**: No (default: `"50000"`)

**Type**: Integer (as string)

**Range**: `1000` - `100000`

**Typical Values**:
- `"10000"`: Small models, memory-constrained
- `"50000"`: Standard (recommended)
- `"100000"`: Large models, high precision

**Effect**: Caps vocabulary size even if target compression not achieved

## Usage Examples

### Example 1: Basic Training from Tabular Preprocessing Output

**Scenario**: Train tokenizer on preprocessed data with default settings

**Input Structure**:
```
/opt/ml/processing/input/
├── train/train_processed_data.csv
├── val/val_processed_data.csv
└── test/test_processed_data.csv
```

**Environment Configuration**:
```bash
export TEXT_FIELD="text"
export TARGET_COMPRESSION="2.5"
export MIN_FREQUENCY="25"
export MAX_VOCAB_SIZE="50000"
```

**Execution**:
```bash
python tokenizer_training.py --job_type training
```

**Expected Output**:
```
Loading training data from /opt/ml/processing/input
Found 3 subdirectories: ['train', 'val', 'test']
Loading and concatenating 3 file(s)...
  Loaded 50,000 texts from train/train_processed_data.csv
  Loaded 10,000 texts from val/val_processed_data.csv
  Loaded 10,000 texts from test/test_processed_data.parquet
Total texts loaded: 70,000

Training BPE tokenizer with compression tuning...
Iteration 1: vocab_size=10000, compression=2.3
Iteration 2: vocab_size=12000, compression=2.48
Iteration 3: vocab_size=11500, compression=2.51
Target compression achieved!

Saving tokenizer artifacts...
Saved tokenizer to /opt/ml/processing/output/tokenizer.json
Saved vocabulary to /opt/ml/processing/output/vocab.json
Saved metadata to /opt/ml/processing/output/tokenizer_metadata.json

Training completed successfully:
  Vocabulary size: 11,500
  Training texts: 70,000
  Target compression: 2.5
```

### Example 2: High Compression for Dense Text

**Scenario**: Train tokenizer with higher compression for word-dense text

**Environment Configuration**:
```bash
export TEXT_FIELD="text"
export TARGET_COMPRESSION="3.0"      # Higher compression
export MIN_FREQUENCY="15"             # Lower threshold for more tokens
export MAX_VOCAB_SIZE="80000"         # Larger vocabulary allowed
```

**Use Case**: When text contains many complete words rather than character sequences

**Expected Outcome**:
- Larger vocabulary size (~18,000 - 25,000)
- Better preservation of word semantics
- More memory usage but better downstream model performance

### Example 3: Memory-Constrained Training

**Scenario**: Train tokenizer with smaller vocabulary for memory-constrained deployment

**Environment Configuration**:
```bash
export TEXT_FIELD="text"
export TARGET_COMPRESSION="2.0"      # Lower compression
export MIN_FREQUENCY="50"             # Higher threshold filters rare tokens
export MAX_VOCAB_SIZE="10000"         # Strict size limit
```

**Use Case**: Deployment on edge devices or memory-limited environments

**Expected Outcome**:
- Smaller vocabulary size (~5,000 - 8,000)
- Lower memory footprint
- Faster inference
- Slightly lower text representation quality

### Example 4: Direct File Input (No Subdirectories)

**Scenario**: Train tokenizer on single data file

**Input Structure**:
```
/opt/ml/processing/input/
└── customer_names.parquet
```

**Environment Configuration**:
```bash
export TEXT_FIELD="customer_name"
export TARGET_COMPRESSION="2.5"
```

**Execution**:
```bash
python tokenizer_training.py --job_type training
```

**Processing**:
- Script detects no subdirectories
- Loads customer_names.parquet directly
- Extracts "customer_name" column
- Trains tokenizer on extracted texts

### Example 5: Multi-Format Mixed Input

**Scenario**: Train tokenizer on mixed file formats in subdirectories

**Input Structure**:
```
/opt/ml/processing/input/
├── train/
│   ├── train_part1.csv
│   └── train_part2.parquet
├── val/
│   └── val_data.tsv
└── test/
    └── test_data.parquet
```

**Environment Configuration**:
```bash
export TEXT_FIELD="text"
export TARGET_COMPRESSION="2.5"
```

**Processing**:
- Discovers 3 subdirectories
- Loads 4 files total (mixed formats)
- Automatically handles CSV, TSV, and Parquet
- Aggregates all texts for training

## Error Handling

### FileNotFoundError: No Data Files Found

**Symptom**: Script fails with no supported data files found

**Error Message**:
```
FileNotFoundError: No supported data files (.parquet, .csv, .tsv) found in /opt/ml/processing/input
```

**Common Causes**:
1. Empty input directory
2. Unsupported file formats (e.g., .txt, .json)
3. Files in nested subdirectories (not directly in train/val/test)

**Resolution**:
- Verify input directory contains supported files
- Check file extensions match .csv, .tsv, or .parquet
- Ensure files are in correct subdirectories

### ValueError: Missing TEXT_FIELD Column

**Symptom**: Script fails during text extraction

**Error Message**:
```
ValueError: Column 'text' not found in data. Available: ['id', 'customer_name', 'label']
```

**Common Causes**:
1. TEXT_FIELD environment variable doesn't match actual column name
2. Column name case mismatch
3. Different preprocessing output than expected

**Resolution**:
```bash
# Check actual column names in your data
export TEXT_FIELD="customer_name"  # Match actual column name
```

### ValueError: No Training Texts Found

**Symptom**: Script fails with empty text list

**Error Message**:
```
ValueError: No training texts found
```

**Common Causes**:
1. All text values are null/missing
2. TEXT_FIELD column exists but is empty
3. Data files loaded but no valid text extracted

**Resolution**:
- Verify data files contain non-null text values
- Check text field contains actual text data
- Inspect sample rows from input files

### MemoryError: Out of Memory During Training

**Symptom**: Script crashes with memory error

**Error Message**:
```
MemoryError: Unable to allocate array
```

**Common Causes**:
1. Too many texts loaded (>1M rows)
2. Instance type with insufficient memory
3. Vocabulary size too large

**Resolution**:
```bash
# Option 1: Reduce vocabulary size
export MAX_VOCAB_SIZE="20000"
export TARGET_COMPRESSION="2.0"

# Option 2: Use larger instance type
# ml.m5.xlarge (16GB) → ml.m5.2xlarge (32GB)

# Option 3: Sample training data
# Modify script to load subset of texts
```

## Best Practices

### For Production Deployments

1. **Use Parquet Format**
   - 5-10x faster loading than CSV
   - Smaller file sizes (better compression)
   - Automatic schema preservation

2. **Leverage Multi-Split Aggregation**
   - Train on ALL data (train + val + test)
   - More diverse vocabulary
   - Better coverage of edge cases
   - Consistent tokenizer across all splits

3. **Set Appropriate Compression Target**
   ```bash
   # For Names3Risk (customer names)
   export TARGET_COMPRESSION="2.5"  # Balanced
   
   # For dense text (product descriptions)
   export TARGET_COMPRESSION="3.0"  # Higher compression
   
   # For character-heavy text (URLs, codes)
   export TARGET_COMPRESSION="2.0"  # Lower compression
   ```

4. **Configure Min Frequency Threshold**
   ```bash
   # Standard (balanced vocabulary)
   export MIN_FREQUENCY="25"
   
   # Small vocabulary (memory-constrained)
   export MIN_FREQUENCY="50"
   
   # Large vocabulary (high precision)
   export MIN_FREQUENCY="10"
   ```

5. **Monitor Vocabulary Size**
   - Check tokenizer_metadata.json after training
   - Verify vocab_size within acceptable range
   - Adjust TARGET_COMPRESSION if needed

### For Development

1. **Start with Small Datasets**
   - Test on subset of data first
   - Verify pipeline works end-to-end
   - Estimate resource requirements

2. **Use CSV for Debugging**
   - Human-readable format
   - Easy to inspect sample data
   - Verify text field extraction

3. **Validate Text Field**
   ```python
   # Before training, check column exists
   import pandas as pd
   df = pd.read_csv("train_data.csv")
   print(df.columns.tolist())
   print(df['text'].head())
   ```

4. **Monitor Training Progress**
   - Watch log output for compression iterations
   - Check if target compression achieved
   - Verify vocabulary size is reasonable

### For Optimization

1. **Choose Right Instance Type**
   | Dataset Size | Recommended Instance | Memory | Cost |
   |--------------|---------------------|--------|------|
   | <50K texts | ml.m5.large | 8GB | Low |
   | 50K-200K | ml.m5.xlarge | 16GB | Medium |
   | 200K-500K | ml.m5.2xlarge | 32GB | Higher |
   | >500K | ml.m5.4xlarge | 64GB | High |

2. **Optimize File Format**
   - Convert CSV to Parquet upstream
   - Reduces training time by 50-70%
   - Smaller storage costs

3. **Balance Vocabulary Size vs Compression**
   - Higher compression = larger vocab = more memory
   - Lower compression = smaller vocab = less precise
   - Find sweet spot for your use case

4. **Cache Tokenizer Artifacts**
   - Reuse trained tokenizer across experiments
   - Avoid retraining on same data
   - Store in model registry or S3

## Integration Patterns

### Upstream Integration (Tabular Preprocessing)

```
TabularPreprocessing
   ↓ (outputs: preprocessed train/val/test with 'text' column)
TokenizerTraining
   ↓ (outputs: tokenizer.json, vocab.json, metadata)
```

**Data Flow**:
1. TabularPreprocessing creates "text" column from source fields
2. TabularPreprocessing splits into train/val/test subdirectories
3. TokenizerTraining loads ALL subdirectories
4. TokenizerTraining trains on combined texts
5. Tokenizer artifacts ready for model training

### Downstream Integration (Model Training)

```
TokenizerTraining
   ↓ (outputs: tokenizer.json)
PyTorchTraining
   ↓ (loads tokenizer for text processing)
```

**Artifact Flow**:
1. TokenizerTraining produces tokenizer.json
2. PyTorchTraining loads tokenizer via HuggingFace API
3. Text data tokenized during model training
4. Consistent vocabulary across training/inference

### Complete Pipeline Example

```
1. DummyDataLoading/CradleDataLoading
   ↓ (raw data)
2. TabularPreprocessing
   ↓ (preprocessed with 'text' column)
3. TokenizerTraining
   ↓ (trained tokenizer)
4. PyTorchTraining
   ↓ (trained model with tokenizer)
5. ModelInference
   ↓ (predictions using tokenizer)
```

## Troubleshooting

### Issue 1: Tokenizer Not Learning Domain Terms

**Symptom**: Vocabulary missing expected domain-specific tokens (e.g., email patterns, product codes)

**Diagnosis**:
```bash
# Check vocab.json for expected tokens
cat /opt/ml/processing/output/vocab.json | grep "@"
cat /opt/ml/processing/output/vocab.json | grep ".com"
```

**Solutions**:
1. Lower MIN_FREQUENCY to include rarer tokens
2. Increase TARGET_COMPRESSION for larger vocabulary
3. Verify training texts contain domain terms

### Issue 2: Training Takes Too Long

**Symptom**: Tokenizer training exceeds expected time

**Diagnosis**:
- Check number of texts loaded
- Monitor compression iteration count
- Verify instance type

**Solutions**:
1. Use Parquet instead of CSV (5-10x faster)
2. Reduce MAX_VOCAB_SIZE if very large
3. Use larger instance type with more CPUs
4. Sample training data if millions of texts

### Issue 3: Inconsistent Tokenization Across Splits

**Symptom**: Different tokenization behavior on train vs test data

**Diagnosis**:
- This should NOT happen with current implementation
- Script trains on ALL splits together

**Verification**:
```python
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_file("tokenizer.json")

# Test on samples from each split
train_sample = "john.smith@example.com"
test_sample = "jane.doe@example.com"

print(tokenizer.encode(train_sample).ids)
print(tokenizer.encode(test_sample).ids)
# Should use same vocabulary
```

### Issue 4: Vocabulary Size Not Meeting Target

**Symptom**: Final vocabulary size far from expected based on compression target

**Diagnosis**:
- Check tokenizer_metadata.json for final vocab_size
- Review training logs for compression iterations

**Solutions**:
1. Adjust TARGET_COMPRESSION (higher = larger vocab)
2. Increase MAX_VOCAB_SIZE if hitting limit
3. Lower MIN_FREQUENCY to include more tokens
4. Verify training data has sufficient text diversity

## References

### Related Scripts

- **Preprocessing Scripts:**
  - [`tabular_preprocessing.py`](tabular_preprocess_script.md): Upstream data preprocessing that creates "text" column
  
- **Training Scripts:**
  - [`pytorch_training.py`](pytorch_training_script.md): Uses trained tokenizer for model training

### Related Documentation

- **Contract**: [`src/cursus/steps/contracts/tokenizer_training_contract.py`](../../src/cursus/steps/contracts/tokenizer_training_contract.py) - Complete contract specification
- **Config**: [`src/cursus/steps/configs/config_tokenizer_training_step.py`](../../src/cursus/steps/configs/config_tokenizer_training_step.py) - Configuration class  
- **Processing Module**: [`src/cursus/processing/custom_tokenizers/bpe_tokenizer.py`](../../src/cursus/processing/custom_tokenizers/bpe_tokenizer.py) - CompressionBPETokenizer implementation

### External References

- **[HuggingFace Tokenizers](https://huggingface.co/docs/tokenizers/)**: Official tokenizers library documentation
- **[BPE Algorithm](https://arxiv.org/abs/1508.07909)**: Original Byte Pair Encoding paper
- **[Subword Tokenization](https://arxiv.org/abs/1808.06226)**: Survey of subword tokenization methods
- **[SentencePiece](https://github.com/google/sentencepiece)**: Alternative tokenization approach (for comparison)

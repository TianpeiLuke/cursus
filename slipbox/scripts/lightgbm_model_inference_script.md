---
tags:
  - code
  - processing_script
  - model_inference
  - lightgbm
  - prediction_generation
keywords:
  - lightgbm inference
  - model prediction
  - preprocessing pipeline
  - format preservation
  - modular inference
  - multi-format output
  - inference-only
topics:
  - model inference
  - prediction generation
  - modular ML pipelines
language: python
date of note: 2025-11-18
---

# LightGBM Model Inference Script Documentation

## Overview

The `lightgbm_model_inference.py` script provides pure inference capabilities for trained LightGBM models, generating predictions without metrics computation or visualization. This modular design enables flexible pipeline architectures where inference results can be cached, reused, and processed by different downstream components.

The script loads trained models with preprocessing artifacts, processes evaluation data through the same preprocessing pipeline used during training, generates predictions, and saves results in multiple formats (CSV, TSV, Parquet, JSON). It preserves all original data columns while adding probability scores for each class, making it ideal for inference-focused workflows that separate prediction generation from evaluation.

Key differentiators from the evaluation script:
- **Inference-only**: No metrics computation, visualization, or comparison analysis
- **Multi-format output**: Supports CSV, TSV, Parquet, and JSON with configurable orientations
- **Modular design**: Enables caching and reuse of inference results
- **Lightweight**: ~600 lines vs ~1,400 lines for full evaluation
- **Flexible integration**: Works as a standalone inference service or pipeline component

## Purpose and Major Tasks

### Primary Purpose
Generate predictions from trained LightGBM models by loading model artifacts, preprocessing input data identically to training, and producing probability scores in multiple output formats suitable for downstream processing.

### Major Tasks

1. **Package Installation**: Dynamic installation of LightGBM from public or secure PyPI based on environment configuration

2. **Model Artifact Loading**: Load complete model package including:
   - Trained LightGBM booster (lightgbm_model.txt)
   - Risk table mappings for categorical features
   - Numerical imputation dictionaries
   - Feature column specifications
   - Model hyperparameters and metadata

3. **Data Loading with Format Detection**: Auto-detect and load input data in CSV, TSV, or Parquet formats

4. **Preprocessing Pipeline Execution**:
   - Apply risk table mapping to categorical features
   - Perform numerical imputation on missing values
   - Ensure features are numeric and model-ready
   - Preserve all original columns (ID, label, metadata)

5. **Prediction Generation**: Generate class probability scores using the trained model for binary or multiclass classification

6. **Multi-Format Output**: Save predictions with original data in user-specified format (CSV, TSV, Parquet, or JSON)

7. **Success Signaling**: Create success and health check markers for pipeline orchestration

## Script Contract

### Entry Point
```
lightgbm_model_inference.py
```

### Input Paths

| Path | Location | Description |
|------|----------|-------------|
| `model_input` | `/opt/ml/processing/input/model` | Model artifacts directory |
| `processed_data` | `/opt/ml/processing/input/eval_data` | Evaluation/inference data |

**Model Input Structure**:
```
/opt/ml/processing/input/model/
├── model.tar.gz (optional, auto-extracted)
├── lightgbm_model.txt          # Trained model
├── risk_table_map.pkl          # Categorical mappings
├── impute_dict.pkl             # Numerical imputation
├── feature_columns.txt         # Feature names/order
└── hyperparameters.json        # Model metadata
```

**Data Input Structure**:
```
/opt/ml/processing/input/eval_data/
└── data.{csv|tsv|parquet}      # Input data with features
```

### Output Paths

| Path | Location | Description |
|------|----------|-------------|
| `eval_output` | `/opt/ml/processing/output/eval` | Predictions with probabilities |

**Output Structure**:
```
/opt/ml/processing/output/eval/
├── predictions.{format}         # Predictions in specified format
├── _SUCCESS                     # Success marker
└── _HEALTH                      # Health check file
```

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ID_FIELD` | Name of ID column in data | `"customer_id"` |
| `LABEL_FIELD` | Name of label column in data | `"is_fraud"` |

### Optional Environment Variables

#### Output Format Configuration

| Variable | Default | Description | Options |
|----------|---------|-------------|---------|
| `OUTPUT_FORMAT` | `"csv"` | Output file format | `"csv"`, `"tsv"`, `"parquet"`, `"json"` |
| `JSON_ORIENT` | `"records"` | JSON orientation | `"records"`, `"index"`, `"values"`, `"split"`, `"table"` |

**JSON Orientation Options**:
- `records`: `[{column -> value}, ..., {column -> value}]` - Array of objects (default)
- `index`: `{index -> {column -> value}}` - Nested dict with index as keys
- `values`: `[[row], [row], ...]` - 2D array of values only
- `split`: `{'index': [...], 'columns': [...], 'data': [...]}` - Split components
- `table`: `{'schema': {...}, 'data': [{...}, {...}, ...]}` - Table schema format

#### Package Installation Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_SECURE_PYPI` | `"true"` | Use secure CodeArtifact PyPI |

### Job Arguments

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `--job_type` | `str` | Yes | Type of inference job (e.g., "inference", "validation") |

### Framework Dependencies

- **lightgbm** >= 3.3.0, <4.0.0 (model framework)
- **pandas** >= 1.3.0 (data processing)
- **numpy** >= 1.21.0 (numerical operations)

## Input Data Structure

### Expected Input Format

```
/opt/ml/processing/input/eval_data/
└── data.{csv|tsv|parquet}
```

**Input file can be**:
- CSV (`.csv`) - Comma-separated values
- TSV (`.tsv`) - Tab-separated values  
- Parquet (`.parquet`) - Columnar binary format

**Format Detection**: Automatic based on file extension

### Required Columns

- **ID Column**: Configurable via `ID_FIELD` environment variable (default: `"id"`)
- **Label Column**: Configurable via `LABEL_FIELD` (default: `"label"`)
- **Feature Columns**: Must match features in `feature_columns.txt` from model artifacts

### Optional Columns

Any additional columns present in the input will be preserved in the output (e.g., metadata, timestamps, auxiliary information).

### Input Data Requirements

1. **Feature Compatibility**: Input features should match those used during training
2. **Missing Features**: Script handles missing features gracefully (uses available features only)
3. **Data Types**: Features can be categorical or numerical (preprocessing handles conversion)
4. **Missing Values**: Automatically imputed using training-time statistics

## Output Data Structure

### Output File Structure

**Default (CSV)**:
```
/opt/ml/processing/output/eval/
├── predictions.csv
├── _SUCCESS
└── _HEALTH
```

**Parquet**:
```
/opt/ml/processing/output/eval/
├── predictions.parquet
├── _SUCCESS
└── _HEALTH
```

**JSON**:
```
/opt/ml/processing/output/eval/
├── predictions.json
├── _SUCCESS
└── _HEALTH
```

### Output Data Format

**Columns in Output**:
- All original input columns (preserved)
- `prob_class_0`: Probability score for class 0
- `prob_class_1`: Probability score for class 1
- ... (additional classes for multiclass)

**Example Output (CSV)**:
```csv
id,feature_1,feature_2,...,label,prob_class_0,prob_class_1
cust_001,1.5,2.3,...,0,0.7234,0.2766
cust_002,2.1,3.4,...,1,0.3145,0.6855
...
```

**Example Output (JSON with records orientation)**:
```json
[
  {
    "id": "cust_001",
    "feature_1": 1.5,
    "feature_2": 2.3,
    "label": 0,
    "prob_class_0": 0.7234,
    "prob_class_1": 0.2766
  },
  {
    "id": "cust_002",
    "feature_1": 2.1,
    "feature_2": 3.4,
    "label": 1,
    "prob_class_0": 0.3145,
    "prob_class_1": 0.6855
  }
]
```

### Success Markers

- **`_SUCCESS`**: Empty file indicating successful completion
- **`_HEALTH`**: Contains timestamp in format `"healthy: 2025-11-18T17:30:00"`

## Key Functions and Tasks

### Package Installation Component

#### `install_packages(packages, use_secure)`
**Purpose**: Install required Python packages from public or secure PyPI

**Algorithm**:
```python
1. Log installation configuration (secure vs public PyPI)
2. IF use_secure:
   - Call install_packages_from_secure_pypi()
3. ELSE:
   - Call install_packages_from_public_pypi()
4. Log completion status
```

**Parameters**:
- `packages` (List[str]): Package specifications (e.g., `["lightgbm>=3.3.0,<4.0.0"]`)
- `use_secure` (bool): Whether to use secure CodeArtifact PyPI

### Model Loading Component

#### `load_model_artifacts(model_dir)`
**Purpose**: Load complete model package with automatic tar.gz extraction

**Algorithm**:
```python
1. Check for model.tar.gz in model_dir
2. IF tar.gz exists AND lightgbm_model.txt missing:
   - Extract model.tar.gz to model_dir
   - Log extraction success
3. Load individual artifacts:
   - lightgbm_model.txt → lgb.Booster object
   - risk_table_map.pkl → Dict (categorical mappings)
   - impute_dict.pkl → Dict (imputation values)
   - feature_columns.txt → List[str] (parse format: "index,name")
   - hyperparameters.json → Dict (model metadata)
4. Return tuple of all artifacts
```

**Returns**: `Tuple[lgb.Booster, Dict, Dict, List[str], Dict]`
- model: LightGBM Booster object
- risk_tables: Categorical feature mappings
- impute_dict: Numerical imputation dictionary
- feature_columns: Ordered list of feature names
- hyperparams: Model hyperparameters and metadata

**Error Handling**:
- Raises `FileNotFoundError` if neither tar.gz nor extracted files found
- Lists available files for debugging

### Data Loading Component

#### `load_eval_data(eval_data_dir)`
**Purpose**: Load evaluation data with automatic format detection

**Algorithm**:
```python
1. Search eval_data_dir for files: .csv, .tsv, .parquet
2. Sort files and select first match
3. Call load_dataframe_with_format():
   - Detect format from extension
   - Load with appropriate reader
4. Return (DataFrame, format_string)
```

**Returns**: `Tuple[pd.DataFrame, str]`

#### `load_dataframe_with_format(file_path)`
**Purpose**: Load DataFrame with format detection

**Algorithm**:
```python
1. Detect format from file extension:
   - .csv → "csv"
   - .tsv → "tsv"
   - .parquet → "parquet"
2. Load with format-specific reader:
   - csv: pd.read_csv(file_path)
   - tsv: pd.read_csv(file_path, sep='\t')
   - parquet: pd.read_parquet(file_path)
3. Return (DataFrame, format)
```

**Time Complexity**: O(n × m) where n=rows, m=columns

### Preprocessing Component

#### `preprocess_inference_data(df, feature_columns, risk_tables, impute_dict)`
**Purpose**: Apply complete preprocessing pipeline maintaining data integrity

**Algorithm**:
```python
1. Copy input DataFrame to preserve original
2. Filter features to available columns:
   available = [f for f in feature_columns if f in df.columns]
3. Apply risk table mapping:
   FOR each feature in risk_tables:
       IF feature in available:
           Create RiskTableMappingProcessor
           Transform feature values to risk scores
4. Apply numerical imputation:
   Create NumericalVariableImputationProcessor
   Transform feature DataFrame
   Update result with imputed values
5. Convert features to numeric (coerce errors → 0)
6. Return preprocessed DataFrame (all columns preserved)
```

**Key Property**: All non-feature columns (ID, label, metadata) are preserved unchanged

**Time Complexity**: O(n × d) where n=rows, d=features

### Prediction Generation Component

#### `generate_predictions(model, df, feature_columns, hyperparams)`
**Purpose**: Generate class probability predictions

**Algorithm**:
```python
1. Filter to available features
2. Extract feature matrix: X = df[available_features].values
3. Generate predictions: y_prob = model.predict(X)
4. Handle output format:
   IF y_prob.ndim == 1:  # Binary case
       Convert to two columns: [[1-p, p], ...]
5. Return prediction array
```

**Returns**: `np.ndarray` of shape (n_samples, n_classes)

**Time Complexity**: O(n × d × trees) for LightGBM prediction

### Output Management Component

#### `save_predictions(df, predictions, output_dir, input_format, ...)`
**Purpose**: Save predictions with original data in specified format

**Algorithm**:
```python
1. Copy input DataFrame
2. Add prediction columns:
   FOR i in range(n_classes):
       output_df[f'prob_class_{i}'] = predictions[:, i]
3. Create output directory if needed
4. IF format == "json":
   - Convert numpy types to native Python types
   - Save with specified JSON orientation
5. ELSE:  # csv, tsv, parquet
   - Call save_dataframe_with_format()
6. Log save location
7. Return output path
```

**JSON Type Conversion**:
- Handles numpy int/float types for JSON serialization
- Preserves object types (strings) as-is

#### `save_dataframe_with_format(df, output_path, format_str)`
**Purpose**: Save DataFrame in specified format

**Algorithm**:
```python
1. Add appropriate extension to output_path
2. Save with format-specific writer:
   - csv: df.to_csv(path, index=False)
   - tsv: df.to_csv(path, sep='\t', index=False)
   - parquet: df.to_parquet(path, index=False)
3. Return final file path
```

### Main Entry Point

#### `main(input_paths, output_paths, environ_vars, job_args)`
**Purpose**: Orchestrate complete inference workflow

**Algorithm**:
```python
1. Extract and validate paths from parameters
2. Extract environment variables (ID_FIELD, LABEL_FIELD, OUTPUT_FORMAT)
3. Create output directories
4. Load model artifacts
5. Load and detect input data format
6. Identify ID and label columns
7. Preprocess data (preserves all columns)
8. Generate predictions
9. Determine final output format:
   - Use OUTPUT_FORMAT if set
   - Otherwise use detected input format
10. Save predictions with original data
11. Create success markers (_SUCCESS, _HEALTH)
12. Exit with code 0
```

**Error Handling**:
- Catches all exceptions
- Creates _FAILURE marker with error message
- Exits with code 1 on failure

## Algorithms and Data Structures

### Algorithm 1: Automatic Model Tar.gz Extraction

**Problem**: Model artifacts may be provided as compressed tar.gz archive or as extracted files

**Solution Strategy**:
1. Check for both tar.gz and expected extracted file
2. Extract only if tar.gz exists AND model file missing
3. Avoid re-extraction if files already present
4. Provide clear error messages if neither found

**Algorithm**:
```python
model_tar_path = os.path.join(model_dir, "model.tar.gz")
model_bst_path = os.path.join(model_dir, "lightgbm_model.txt")

IF exists(model_tar_path) AND NOT exists(model_bst_path):
    # Extract needed
    WITH tarfile.open(model_tar_path, "r:gz") as tar:
        tar.extractall(path=model_dir)
    LOG("✓ Extracted model.tar.gz")
    
ELSE IF exists(model_bst_path):
    # Already extracted
    LOG("Found extracted artifacts - using directly")
    
ELSE:
    # Error case
    available_files = listdir(model_dir)
    RAISE FileNotFoundError(
        f"Model artifacts not found. Available: {available_files}"
    )
```

**Complexity**: O(1) for file checks, O(size) for extraction

### Algorithm 2: Format-Preserving Data Pipeline

**Problem**: Maintain input data format through processing pipeline while adding predictions

**Solution Strategy**:
1. Detect input format during load
2. Preserve format string through processing
3. Use detected format for output (unless overridden)
4. Support format override via environment variable

**Algorithm**:
```python
# Load phase
df, input_format = load_dataframe_with_format(input_file)
# input_format ∈ {"csv", "tsv", "parquet"}

# Processing phase
df_processed = preprocess_inference_data(df, ...)
# Format information preserved externally

# Prediction phase
predictions = generate_predictions(model, df_processed, ...)

# Save phase
final_format = OUTPUT_FORMAT or input_format
save_predictions(df_processed, predictions, ..., format=final_format)
```

**Benefits**:
- Zero configuration needed for format preservation
- Explicit override available when needed
- Consistent with data pipeline best practices

### Algorithm 3: Feature Availability Handling

**Problem**: Input data may not contain all features used during training

**Solution Strategy**:
1. Filter feature list to available columns
2. Use only available features for preprocessing and prediction
3. Model adapts to available feature subset
4. Log feature availability for monitoring

**Algorithm**:
```python
# Expected features from training
feature_columns = ["f1", "f2", "f3", "f4", "f5"]

# Available features in input data
input_columns = ["id", "f1", "f2", "f4", "label"]

# Compute available features
available_features = [
    f for f in feature_columns 
    if f in input_columns
]
# Result: ["f1", "f2", "f4"]

LOG(f"Using {len(available_features)}/{len(feature_columns)} features")

# Use only available features
X = df[available_features].values
predictions = model.predict(X)
```

**Graceful Degradation**: Model uses available features without error

**Time Complexity**: O(d) where d = number of features

### Data Structure 1: Embedded Processor Classes

The script embeds complete preprocessor implementations to remove external dependencies:

```python
class RiskTableMappingProcessor:
    """
    Maps categorical values to continuous risk scores.
    
    Attributes:
        column_name: str - Feature to process
        risk_tables: Dict - Mapping structure
            {
                "bins": {
                    "value_a": 0.15,
                    "value_b": 0.82,
                    ...
                },
                "default_bin": 0.45
            }
        is_fitted: bool - Ready for use
    """

class NumericalVariableImputationProcessor:
    """
    Imputes missing numerical values.
    
    Attributes:
        imputation_dict: Dict[str, float]
            {
                "feature_1": 42.5,  # mean/median/mode
                "feature_2": 18.3,
                ...
            }
        strategy: str - Imputation method
        is_fitted: bool - Ready for use
    """
```

**Design Rationale**: 
- Removes dependency on external preprocessing package
- Ensures consistent preprocessing across environments
- Simplifies deployment and portability

### Data Structure 2: Model Artifacts Package

```python
ModelArtifacts = Tuple[
    lgb.Booster,              # Trained model
    Dict[str, Dict],          # Risk tables
    Dict[str, float],         # Imputation dict
    List[str],                # Feature columns
    Dict[str, Any]            # Hyperparameters
]

# Risk tables structure
risk_tables: Dict[str, Dict] = {
    "categorical_feature": {
        "bins": {
            "category_a": 0.15,
            "category_b": 0.82,
        },
        "default_bin": 0.45
    },
    ...
}

# Imputation dictionary structure
impute_dict: Dict[str, float] = {
    "numerical_feature_1": 42.5,
    "numerical_feature_2": 18.3,
    ...
}

# Feature columns structure
feature_columns: List[str] = [
    "feature_1",
    "feature_2",
    ...
]

# Hyperparameters structure
hyperparams: Dict[str, Any] = {
    "is_binary": True,
    "num_class": 1,
    "objective": "binary",
    ...
}
```

## Performance Characteristics

### Processing Time

| Operation | Typical Time (10K samples, 100 features) |
|-----------|------------------------------------------|
| Package installation | 5-10 seconds (first run only) |
| Model artifact loading | 1-2 seconds |
| Tar.gz extraction | 0.5-1 second (if needed) |
| Data loading (CSV) | 0.5-1 second |
| Data loading (Parquet) | 0.2-0.5 seconds |
| Risk table mapping | 1-2 seconds |
| Numerical imputation | 0.5-1 second |
| Prediction generation | 1-2 seconds |
| Save predictions (CSV) | 0.5-1 second |
| Save predictions (Parquet) | 0.3-0.7 seconds |
| Save predictions (JSON) | 1-2 seconds |
| **Total (CSV workflow)** | **6-11 seconds** |
| **Total (Parquet workflow)** | **5-9 seconds** |

### Memory Usage

**Typical Memory Profile**:
```
Model artifacts: 10-50 MB (LightGBM model size)
Input data: n × d × 8 bytes (float64)
Preprocessed data: n × d × 8 bytes (copy for preprocessing)
Predictions: n × num_classes × 8 bytes
Output data: n × (d + num_classes) × 8 bytes
Peak: ~3× input data size for 100K samples, 100 features
```

**Memory Efficiency**:
- Parquet format: 40-60% size reduction vs CSV
- JSON format: 20-40% size increase vs CSV (due to field names)
- In-memory: Single copy of data during preprocessing

### Scalability

| Dataset Size | Memory (est.) | Time (est.) |
|--------------|---------------|-------------|
| 1K samples | ~10 MB | 2-3 seconds |
| 10K samples | ~50 MB | 6-11 seconds |
| 100K samples | ~400 MB | 30-60 seconds |
| 1M samples | ~4 GB | 5-10 minutes |

**Recommendation**: For datasets > 1M samples, consider:
- Batch processing in chunks
- Using Parquet format for efficiency
- Distributed processing frameworks

## Error Handling

### Error Types

#### Input Validation Errors

**Missing Model Artifacts**:
- **Cause**: Neither `model.tar.gz` nor extracted files found
- **Handling**: Raises `FileNotFoundError` with available files listed
- **Response**: Script exits with code 1, creates `_FAILURE` marker

**Unsupported File Format**:
- **Cause**: Input file has unsupported extension
- **Handling**: Raises `RuntimeError` with supported formats
- **Response**: Clear error message listing `.csv`, `.tsv`, `.parquet`

**Missing Required Columns**:
- **Cause**: ID or label fields not found in data
- **Handling**: Falls back to first/second columns with warning
- **Response**: Logs column selection for verification

#### Processing Errors

**Risk Table Processing Failure**:
- **Cause**: Malformed risk table dictionary
- **Handling**: Validation in `RiskTableMappingProcessor.__init__`
- **Response**: Detailed error about dict structure requirements

**Imputation Processing Failure**:
- **Cause**: Invalid imputation dictionary values
- **Handling**: Validation in `NumericalVariableImputationProcessor.__init__`
- **Response**: Specific error identifying problematic key-value pair

**Prediction Generation Failure**:
- **Cause**: Feature mismatch or model incompatibility
- **Handling**: Gracefully uses available features only
- **Response**: Logs feature availability, proceeds with subset

#### Output Errors

**Disk Space Exhaustion**:
- **Cause**: Insufficient space for output file
- **Handling**: Exception caught in main(), logged
- **Response**: Creates `_FAILURE` marker with error details

**Permission Errors**:
- **Cause**: Cannot write to output directory
- **Handling**: Exception caught in main(), logged
- **Response**: Creates `_FAILURE` marker with permission error

### Error Response Structure

**Failure Marker Content** (`_FAILURE`):
```
Error: [Exception type and message]
```

**Example**:
```
Error: FileNotFoundError: Model artifacts not found in /opt/ml/processing/input/model. 
Expected either 'model.tar.gz' or 'lightgbm_model.txt'. 
Available files: ['README.md', 'config.json']
```

## Best Practices

### For Production Deployments

1. **Use Parquet Format for Large Datasets**
   - Set `OUTPUT_FORMAT=parquet` for 40-60% size reduction
   - Faster I/O compared to CSV
   - Better compression and column-oriented storage

2. **Monitor Feature Availability**
   - Check logs for feature availability warnings
   - Ensure input data contains expected features
   - Validate prediction quality when features are missing

3. **Implement Output Validation**
   - Verify `_SUCCESS` marker exists before consuming predictions
   - Check `_HEALTH` timestamp for staleness detection
   - Validate prediction probability ranges [0, 1]

4. **Use Appropriate JSON Orientation**
   - `records`: Best for REST APIs (default)
   - `table`: Best for schema-aware consumers
   - `values`: Best for minimal payload size

### For Development

1. **Test with Small Datasets First**
   - Validate preprocessing pipeline with sample data
   - Check output format and structure
   - Verify feature handling behavior

2. **Use CSV for Debugging**
   - Human-readable output
   - Easy to inspect in spreadsheet tools
   - Quick validation of predictions

3. **Log Level Configuration**
   - Default INFO level provides good visibility
   - Increase to DEBUG for detailed preprocessing info
   - Reduce to WARNING for production

### For Performance Optimization

1. **Choose Efficient Formats**
   - Parquet: Best for storage and I/O efficiency
   - CSV: Best for human readability and compatibility
   - JSON: Best for API integration

2. **Batch Processing for Large Datasets**
   - Split large datasets into chunks
   - Process chunks independently
   - Combine outputs if needed

3. **Minimize Data Copies**
   - Script already optimized with single preprocessing copy
   - Avoid unnecessary DataFrame operations
   - Use in-place operations where possible

## Example Configurations

### Example 1: Standard CSV Inference
```bash
# Environment variables
export ID_FIELD="customer_id"
export LABEL_FIELD="is_fraud"
export OUTPUT_FORMAT="csv"  # Default, can be omitted
export USE_SECURE_PYPI="true"

# Execution
python lightgbm_model_inference.py --job_type inference
```

**Use Case**: Standard inference workflow with CSV input and output

### Example 2: Parquet High-Performance Pipeline
```bash
# Environment variables  
export ID_FIELD="transaction_id"
export LABEL_FIELD="label"
export OUTPUT_FORMAT="parquet"  # Efficient binary format
export USE_SECURE_PYPI="true"

# Execution
python lightgbm_model_inference.py --job_type inference
```

**Use Case**: Large dataset processing with optimal storage efficiency

### Example 3: JSON API Integration
```bash
# Environment variables
export ID_FIELD="request_id"
export LABEL_FIELD="target"
export OUTPUT_FORMAT="json"
export JSON_ORIENT="records"  # Array of objects format
export USE_SECURE_PYPI="false"  # Public PyPI

# Execution
python lightgbm_model_inference.py --job_type inference
```

**Use Case**: Integration with REST APIs or microservices expecting JSON

### Example 4: Table Schema JSON for Data Consumers
```bash
# Environment variables
export ID_FIELD="id"
export LABEL_FIELD="label"
export OUTPUT_FORMAT="json"
export JSON_ORIENT="table"  # Includes schema information
export USE_SECURE_PYPI="true"

# Execution
python lightgbm_model_inference.py --job_type validation
```

**Use Case**: Schema-aware data consumers requiring type information

## Integration Patterns

### Upstream Integration

```
TabularPreprocessing
   ↓ (outputs: preprocessed_data)
LightGBMTraining
   ↓ (outputs: model_artifacts)
LightGBMModelInference
   ↓ (outputs: predictions with probabilities)
```

**Data Flow**:
1. TabularPreprocessing creates feature-engineered dataset
2. LightGBMTraining produces model.tar.gz with artifacts
3. LightGBMModelInference generates predictions

### Downstream Integration

```
LightGBMModelInference
   ↓ (outputs: predictions.csv with prob_class_*)
   ├→ ModelMetricsComputation (metrics)
   ├→ ModelCalibration (calibrated scores)
   └→ PercentileModelCalibration (percentile mapping)
```

**Output Compatibility**:
- ModelMetricsComputation expects: ID column + label column + `prob_class_*` columns
- ModelCalibration expects: Label column + `prob_class_*` columns
- PercentileModelCalibration expects: `prob_class_*` columns

### Modular Pipeline Pattern

```
Pipeline A (Training):
  TabularPreprocessing → LightGBMTraining → (cache model)

Pipeline B (Inference):
  (load cached model) → LightGBMModelInference → (cache predictions)
  
Pipeline C (Evaluation):
  (load cached predictions) → ModelMetricsComputation → Reports

Pipeline D (Calibration):
  (load cached predictions) → ModelCalibration → Calibrated Model
```

**Benefits**:
- **Decoupling**: Inference separate from evaluation
- **Caching**: Reuse predictions across multiple evaluations
- **Flexibility**: Different evaluation strategies without re-inference
- **Cost efficiency**: Avoid redundant model loading and prediction

### Workflow Example

**Complete ML Pipeline**:
```
1. Data Preparation
   TabularPreprocessing (job_type=training)
   → preprocessed_train.csv
   
2. Model Training
   LightGBMTraining
   → model.tar.gz
   
3. Inference on Test Data
   LightGBMModelInference
   → predictions.csv (with prob_class_0, prob_class_1)
   
4. Parallel Downstream Processing
   ├→ ModelMetricsComputation → metrics.json
   ├→ ModelCalibration → calibrated_model
   └→ PercentileModelCalibration → percentile_mapping.json
```

## Troubleshooting

### Issue 1: Model Artifacts Not Found

**Symptom**:
```
FileNotFoundError: Model artifacts not found in /opt/ml/processing/input/model.
Expected either 'model.tar.gz' or 'lightgbm_model.txt'.
Available files: []
```

**Common Causes**:
1. Model path misconfigured in pipeline
2. Model training step failed to produce output
3. S3 path incorrect in ProcessingInput

**Solutions**:
1. Verify model training step completed successfully
2. Check S3 path in model input configuration
3. Ensure model.tar.gz was uploaded to correct location
4. Review training step logs for model saving errors

### Issue 2: Feature Mismatch Warning

**Symptom**:
```
WARNING: Found 85 out of 100 expected feature columns
```

**Common Causes**:
1. Input data missing some features
2. Feature engineering step produced different features
3. Column name changes between training and inference

**Solutions**:
1. Verify preprocessing pipeline matches training
2. Check feature_columns.txt from model artifacts
3. Ensure consistent column naming convention
4. Review preprocessing logs for dropped features

### Issue 3: Output Format Not Recognized

**Symptom**:
```
RuntimeError: Unsupported file format: .xlsx
```

**Common Causes**:
1. Input file has unsupported format
2. File extension doesn't match content
3. Corrupted input file

**Solutions**:
1. Convert input to CSV, TSV, or Parquet
2. Verify file extension matches actual format
3. Check file integrity and re-upload if needed
4. Use format conversion tools before inference

### Issue 4: Memory Error on Large Dataset

**Symptom**:
```
MemoryError: Unable to allocate array with shape (1000000, 100)
```

**Common Causes**:
1. Dataset too large for available memory
2. Multiple data copies in memory
3. Insufficient instance resources

**Solutions**:
1. Use Parquet format for better memory efficiency
2. Process data in batches/chunks
3. Increase instance size for processing
4. Consider distributed processing approach

### Issue 5: JSON Serialization Error

**Symptom**:
```
TypeError: Object of type 'float64' is not JSON serializable
```

**Common Causes**:
1. Numpy types not converted to native Python types
2. Special values (NaN, Inf) in predictions
3. JSON orient parameter misconfigured

**Solutions**:
1. Script automatically handles numpy conversion (check version)
2. Verify no NaN/Inf values in predictions
3. Try different JSON_ORIENT settings
4. Consider using CSV or Parquet format instead

## References

### Related Scripts

- **[`lightgbm_model_eval.py`](./lightgbm_model_eval_script.md)**: Comprehensive evaluation script with metrics and visualizations
- **[`xgboost_model_inference.py`](./xgboost_model_inference_script.md)**: Similar inference script for XGBoost models
- **[`lightgbm_training.py`](./lightgbm_training_script.md)**: Upstream training script that produces model artifacts

### Related Documentation

- **Contract**: `src/cursus/steps/contracts/lightgbm_model_inference_contract.py`
- **Step Specification**: LightGBMModelInference step specification
- **Config Class**: `src/cursus/steps/configs/config_lightgbm_model_inference_step.py`
- **Builder**: `src/cursus/steps/builders/builder_lightgbm_model_inference_step.py`

### Related Design Documents

No specific design documents currently exist for this script. General patterns apply from:
- **[Data Format Preservation Patterns](../1_design/data_format_preservation_patterns.md)**: Format detection and preservation strategy
- **[Processing Step Builder Patterns](../1_design/processing_step_builder_patterns.md)**: General processing step architecture

### External References

- **LightGBM Documentation**: https://lightgbm.readthedocs.io/en/latest/
- **Pandas DataFrame Documentation**: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html
- **Parquet Format**: https://parquet.apache.org/docs/
- **JSON Orient Parameter**: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html

---

## Document Metadata

**Author**: Cursus Framework Team  
**Last Updated**: 2025-11-18  
**Script Version**: 2025-11-18  
**Documentation Version**: 1.0  
**Review Status**: Complete

**Change Log**:
- 2025-11-18: Initial comprehensive documentation created
- 2025-11-18: LightGBM model inference script implemented

**Related Scripts**: 
- Upstream: `lightgbm_training.py`
- Related: `lightgbm_model_eval.py`, `xgboost_model_inference.py`
- Downstream: `model_metrics_computation.py`, `model_calibration.py`

---
tags:
  - code
  - implementation
  - processing_script
  - payload_generation
  - mims
  - testing
  - multi_modal
keywords:
  - MIMS payload
  - inference testing
  - payload generation
  - sample data
  - model deployment
  - hyperparameters
  - content types
  - multi-modal models
  - custom payloads
topics:
  - model deployment
  - testing workflows
  - MIMS integration
  - multi-modal ML
language: python
date of note: 2025-11-29
---

# Payload Script Documentation

## Overview

The `payload.py` script implements the MIMS (Model Inference Management System) payload generation workflow that creates sample inference payloads for testing deployed models. The script extracts field information from model hyperparameters and generates properly formatted test payloads in multiple content types (JSON, CSV) with configurable default values.

**NEW in 2025-11-29**: This script now supports **multi-modal models** (tabular, bimodal, trimodal), **custom user-provided payloads**, **intelligent text field generation**, and **robust validation** with feature_columns.txt integration for XGBoost/LightGBM models.

This script serves as a critical component in the model deployment testing pipeline, enabling automated generation of sample inference requests that match the exact input schema expected by the deployed model. It intelligently handles numeric, text, and multi-modal fields (text + tabular combinations), supports custom field values through multiple configuration methods, and produces archived payloads ready for inference endpoint testing.

The payload generation process is driven by hyperparameters embedded in the model artifacts and optionally by feature_columns.txt for XGBoost/LightGBM models, ensuring that generated payloads always match the model's expected input format and field ordering. This eliminates manual payload creation and reduces deployment testing errors.

## Purpose and Major Tasks

### Primary Purpose
Generate sample inference payloads from model hyperparameters in multiple content formats (JSON, CSV) for testing MIMS-deployed model endpoints with properly structured input data. Supports tabular-only, bimodal (text + tabular), and trimodal (dual text + tabular) models with optional custom user-provided payloads.

### Major Tasks

1. **Hyperparameter Extraction**: Extract and parse hyperparameters.json from model artifacts (tar.gz or directory) with multiple fallback strategies for robust discovery

2. **Model Type Detection**: Automatically detect model type (tabular, bimodal, trimodal) from hyperparameters to determine required fields and structure

3. **Source of Truth Identification**: Use feature_columns.txt (XGBoost/LightGBM) or hyperparameters.json (PyTorch) as authoritative source for field requirements and ordering

4. **Variable List Creation**: Create typed variable lists from field information distinguishing numeric (tabular) and text (categorical) fields while excluding label and ID columns

5. **Custom Payload Loading**: Optionally load user-provided custom payloads from JSON/CSV/Parquet files or directories for testing with real data

6. **Environment Configuration**: Parse environment variables for content types, default values, and field-specific configurations with validation and error handling

7. **Intelligent Text Generation**: Generate contextually appropriate text samples using 3-tier priority (user-provided > pattern-based > generic) for realistic test data

8. **Multi-Format Payload Generation**: Generate payloads in multiple content types (application/json, text/csv) with correct field ordering (critical for XGBoost/LightGBM CSV inputs)

9. **Comprehensive Validation**: Validate generated and custom payloads against model requirements with detailed field mapping logs

10. **Payload File Creation**: Save generated payloads to files with descriptive names and proper formatting for each content type

11. **Archive Creation**: Create compressed tar.gz archive containing all payload samples with compression statistics and verification

12. **Logging and Verification**: Provide comprehensive logging including payload samples, validation results, file statistics, and archive creation details for debugging and validation

## Script Contract

### Entry Point
```
payload.py
```

### Input Paths

| Path | Location | Description |
|------|----------|-------------|
| `model_input` | `/opt/ml/processing/input/model` | Model artifacts directory containing hyperparameters.json (may be in tar.gz or loose) and optionally feature_columns.txt |
| `custom_payload_input` (optional) | `/opt/ml/processing/input/custom_payload` | Optional directory or file containing user-provided custom payload sample (JSON/CSV/Parquet) |

### Output Paths

| Path | Location | Description |
|------|----------|-------------|
| `output_dir` | `/opt/ml/processing/output` | Output directory where payload.tar.gz archive is created |

### Required Environment Variables

None - All environment variables have defaults. The script can run with no environment variables set.

### Optional Environment Variables

#### Payload Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTENT_TYPES` | `"application/json"` | Comma-separated list of content types to generate payloads for (e.g., "application/json,text/csv") |
| `DEFAULT_NUMERIC_VALUE` | `"0.0"` | Default value for numeric/tabular fields in generated payloads |
| `DEFAULT_TEXT_VALUE` | `"Sample text for inference testing"` | Default value for text/categorical fields in generated payloads |
| `FIELD_DEFAULTS` | None | **NEW**: JSON dictionary mapping field names to default values (e.g., `'{"field1": "value1", "field2": "value2"}'`) - preferred method for batch field configuration |
| `SPECIAL_FIELD_<fieldname>` | None | Custom value for specific field (e.g., `SPECIAL_FIELD_timestamp="{timestamp}"`) - supports {timestamp} placeholder, overrides FIELD_DEFAULTS for that field |

#### Working Directory Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKING_DIRECTORY` | `"/tmp/mims_payload_work"` | Temporary directory for payload generation and intermediate files |

#### Field Configuration Priority

When multiple configuration methods are used, the priority is:
1. **SPECIAL_FIELD_*** (highest - per-field override)
2. **FIELD_DEFAULTS** (JSON dictionary - batch configuration)
3. **Intelligent defaults** (pattern-based for text fields)
4. **Generic defaults** (DEFAULT_NUMERIC_VALUE, DEFAULT_TEXT_VALUE)

### Job Arguments

No command-line arguments are required. The script uses the standardized contract-based path structure.

## Multi-Modal Model Support

### Model Types

The script automatically detects and supports three model types:

#### 1. Tabular Models (Traditional)
- **Detection**: No text field hyperparameters
- **Fields**: Only numeric and categorical tabular features
- **Examples**: XGBoost, LightGBM, standard tree-based models
- **Payload Structure**: 
  ```json
  {
    "numeric_feature1": "0.0",
    "numeric_feature2": "0.0",
    "categorical_feature": "DEFAULT_TEXT"
  }
  ```

#### 2. Bimodal Models (Text + Tabular)
- **Detection**: `text_name` hyperparameter present
- **Fields**: Single text field + numeric/categorical features
- **Examples**: BERT + tabular, text embeddings + features
- **Payload Structure**:
  ```json
  {
    "id": "TEST_ID_20251129_211500",
    "text_name": "Sample text for inference testing",
    "numeric_feature1": "0.0",
    "categorical_feature": "DEFAULT_TEXT"
  }
  ```

#### 3. Trimodal Models (Dual Text + Tabular)
- **Detection**: `primary_text_name` AND `secondary_text_name` hyperparameters present
- **Fields**: Two text fields + numeric/categorical features
- **Examples**: Product title + description + features, buyer-seller text pairs
- **Payload Structure**:
  ```json
  {
    "id": "TEST_ID_20251129_211500",
    "primary_text_name": "Hello, I need help with my order. Can you assist me?",
    "secondary_text_name": "Package shipped|In transit|Delivered",
    "numeric_feature1": "0.0",
    "categorical_feature": "DEFAULT_TEXT"
  }
  ```

### Field Ordering

**CRITICAL for XGBoost/LightGBM CSV Inputs:**

The script ensures correct field ordering by using the authoritative source:
- **XGBoost/LightGBM**: Uses `feature_columns.txt` (extracted from model.tar.gz if needed)
- **PyTorch Models**: Uses hyperparameters with order: ID → text fields → tabular fields

**Why This Matters:**
- XGBoost/LightGBM expect features in the exact order from training
- CSV format has no field names, only positional values
- Incorrect ordering causes inference failures or wrong predictions
- The script automatically handles extraction and ordering

## Custom Payload Support

### Overview

Users can provide custom payload samples instead of auto-generated ones. This is useful for:
- Testing with real production data samples
- Validating specific edge cases
- Using domain-specific realistic values

### Supported Formats

| Format | Extensions | Loading Method |
|--------|-----------|----------------|
| JSON | `.json` | Direct JSON parsing |
| CSV | `.csv` | First row as dictionary (requires pandas) |
| Parquet | `.parquet` | First row as dictionary (requires pandas) |
| Directory | N/A | Auto-detects and loads first matching file (priority: JSON > CSV > Parquet) |

### Custom Payload Location

Place custom payload at: `/opt/ml/processing/input/custom_payload`

This can be:
- A single file: `custom_payload/sample.json`
- A directory with files: `custom_payload/*.json` (uses first match)

### Validation

Custom payloads are validated against model requirements:
- ✅ **Checks for all required fields** (from feature_columns.txt or hyperparameters)
- ✅ **Validates field completeness** for the detected model type
- ✅ **Logs detailed field mapping** for debugging
- ❌ **Fails with clear error** if required fields are missing

### Example Custom Payload

**File**: `/opt/ml/processing/input/custom_payload/real_sample.json`
```json
{
  "id": "CUSTOMER_12345",
  "chat_history": "I ordered item X but received item Y. Need refund.",
  "shiptrack_events": "Order placed|Payment confirmed|Shipped",
  "days_since_order": "5.0",
  "order_value": "129.99",
  "customer_segment": "premium"
}
```

**Validation Process:**
1. Loads custom payload
2. Detects model type (trimodal in this case)
3. Validates presence of: `id`, `chat_history`, `shiptrack_events`, all tabular features
4. Generates CSV with correct field ordering for XGBoost/LightGBM
5. Creates payload.tar.gz with validated samples

## Intelligent Text Field Generation

### 3-Tier Priority System

Text fields are generated using a hierarchical approach:

#### Priority 1: User-Provided Values
- From `FIELD_DEFAULTS` JSON dictionary
- From `SPECIAL_FIELD_*` environment variables (overrides FIELD_DEFAULTS)
- Exact match or case-insensitive fallback
- Supports `{timestamp}` template placeholder

#### Priority 2: Intelligent Pattern-Based Defaults
Recognizes common field name patterns and provides contextually appropriate defaults:

| Pattern | Example Names | Generated Text |
|---------|---------------|----------------|
| Chat/Dialogue | `chat`, `conversation`, `dialogue` | "Hello, I need help with my order. Can you assist me?" |
| Tracking | `shiptrack`, `event`, `tracking` | "Package shipped\|In transit\|Delivered" |
| Description | `description`, `desc` | "Product description text for testing purposes" |
| Comment | `comment`, `note` | "Additional notes and comments for testing" |
| Title | `title`, `subject` | "Sample title for testing" |
| Message | `message`, `msg` | "Sample message content for testing" |

#### Priority 3: Generic Default
- Falls back to `DEFAULT_TEXT_VALUE` environment variable
- Ultimate fallback: "Sample text for inference testing"

### Configuration Examples

**Example 1: Using FIELD_DEFAULTS (preferred for multiple fields)**
```bash
export FIELD_DEFAULTS='{"chat_history": "Customer service inquiry", "product_title": "Wireless Headphones"}'
```

**Example 2: Using SPECIAL_FIELD_* (backward compatible, overrides FIELD_DEFAULTS)**
```bash
export SPECIAL_FIELD_timestamp="{timestamp}"
export SPECIAL_FIELD_user_id="test_user_12345"
```

**Example 3: Combined approach**
```bash
export FIELD_DEFAULTS='{"chat": "Hello", "description": "Test product"}'
export SPECIAL_FIELD_timestamp="{timestamp}"  # Overrides FIELD_DEFAULTS if "timestamp" key exists there
```

**Result for field "timestamp"**: Uses SPECIAL_FIELD_* value (2025-11-29 21:15:30)
**Result for field "chat"**: Uses FIELD_DEFAULTS value ("Hello")

## Input Data Structure

### Model Input Structure

The script accepts hyperparameters in multiple locations (tried in order):

**Location 1: Standard tar.gz file**
```
/opt/ml/processing/input/model/
└── model.tar.gz
    ├── hyperparameters.json
    └── feature_columns.txt (optional, for XGBoost/LightGBM)
```

**Location 2: Directory named model.tar.gz**
```
/opt/ml/processing/input/model/
└── model.tar.gz/              # Actually a directory
    ├── hyperparameters.json
    └── feature_columns.txt (optional)
```

**Location 3: Direct files in model directory**
```
/opt/ml/processing/input/model/
├── hyperparameters.json
└── feature_columns.txt (optional)
```

**Location 4: Recursive search**
```
/opt/ml/processing/input/model/
└── subdirectory/
    ├── hyperparameters.json
    └── feature_columns.txt (optional)
```

### Hyperparameters Structure

#### Tabular Model (Traditional)
```json
{
  "full_field_list": ["id", "feature1", "feature2", "category1", "label"],
  "tab_field_list": ["feature1", "feature2"],
  "cat_field_list": ["category1"],
  "label_name": "label",
  "id_name": "id"
}
```

#### Bimodal Model (Text + Tabular)
```json
{
  "full_field_list": ["id", "text_field", "feature1", "category1", "label"],
  "tab_field_list": ["feature1"],
  "cat_field_list": ["category1"],
  "text_name": "text_field",
  "label_name": "label",
  "id_name": "id"
}
```

#### Trimodal Model (Dual Text + Tabular)
```json
{
  "full_field_list": ["id", "chat_history", "shiptrack_events", "feature1", "category1", "label"],
  "tab_field_list": ["feature1"],
  "cat_field_list": ["category1"],
  "primary_text_name": "chat_history",
  "secondary_text_name": "shiptrack_events",
  "model_class": "TrimodalBERT",
  "label_name": "label",
  "id_name": "id"
}
```

**Field Definitions:**
- `full_field_list`: Complete list of all fields in the dataset
- `tab_field_list`: List of numeric/tabular fields
- `cat_field_list`: List of categorical/text fields (for tabular models)
- `text_name`: Single text field name (bimodal models)
- `primary_text_name`: First text field name (trimodal models)
- `secondary_text_name`: Second text field name (trimodal models)
- `model_class`: Optional model class name (helps with detection)
- `label_name`: Name of the label column (excluded from payloads)
- `id_name`: Name of the ID column (included in payloads)

**Note**: Only fields in `tab_field_list` or `cat_field_list` are included in payloads for tabular models. Multi-modal models additionally include text fields.

### Feature Columns File (XGBoost/LightGBM)

**File**: `feature_columns.txt`
**Format**: CSV with index and column name
```
0,feature1
1,feature2
2,category1
```

**Purpose**: 
- Defines exact feature ordering for XGBoost/LightGBM models
- Critical for CSV payload generation (positional values)
- Takes precedence over hyperparameters for field ordering

## Output Data Structure

### Working Directory Structure (Temporary)

```
/tmp/mims_payload_work/
├── hyperparameters.json         # Extracted hyperparameters
└── payload_sample/              # Sample payload files
    ├── payload_application_json_0.json
    └── payload_text_csv_0.csv
```

### Output Directory Structure

```
/opt/ml/processing/output/
└── payload.tar.gz               # Compressed archive of payload samples
```

### Payload.tar.gz Contents

When extracted, the tar.gz file contains:

```
payload_application_json_0.json
payload_text_csv_0.csv
```

### Payload Format Examples

#### Tabular Model

**JSON Format**
```json
{
  "feature1": "0.0",
  "feature2": "0.0",
  "category1": "DEFAULT_TEXT"
}
```

**CSV Format**
```csv
0.0,0.0,DEFAULT_TEXT
```

#### Bimodal Model

**JSON Format**
```json
{
  "id": "TEST_ID_20251129_211500",
  "text_name": "Sample text for inference testing",
  "feature1": "0.0",
  "category1": "DEFAULT_TEXT"
}
```

**CSV Format**
```csv
TEST_ID_20251129_211500,Sample text for inference testing,0.0,DEFAULT_TEXT
```

#### Trimodal Model

**JSON Format**
```json
{
  "id": "TEST_ID_20251129_211500",
  "chat_history": "Hello, I need help with my order. Can you assist me?",
  "shiptrack_events": "Package shipped|In transit|Delivered",
  "feature1": "0.0",
  "category1": "DEFAULT_TEXT"
}
```

**CSV Format**
```csv
TEST_ID_20251129_211500,Hello, I need help with my order. Can you assist me?,Package shipped|In transit|Delivered,0.0,DEFAULT_TEXT
```

**Note**: CSV format field ordering is critical for XGBoost/LightGBM and is automatically handled using feature_columns.txt

### Special Field Example

With intelligent defaults:

```json
{
  "id": "TEST_ID_20251129_211500",
  "timestamp": "2025-11-29 21:15:30",
  "chat_history": "Hello, I need help with my order. Can you assist me?",
  "shiptrack_events": "Package shipped|In transit|Delivered",
  "description": "Product description text for testing purposes",
  "feature1": "0.0"
}
```

## Key Functions and Tasks

### Multi-Modal Support Component

#### `detect_model_type(hyperparams: Dict) -> str`
**Purpose**: Automatically detect model type from hyperparameters

**Algorithm**:
```python
1. Get model_class from hyperparams (default: "")
2. Convert to lowercase for comparison
3. Check for trimodal indicators:
   IF "trimodal" in model_class OR
      (primary_text_name exists AND secondary_text_name exists):
      RETURN "trimodal"
4. Check for bimodal indicators:
   IF "multimodal" in model_class OR text_name exists:
      RETURN "bimodal"
5. Default to tabular:
   RETURN "tabular"
```

**Returns**: `str` - One of: "trimodal", "bimodal", "tabular"

**Detection Logic**:
- Trimodal: Dual text fields (primary_text_name + secondary_text_name)
- Bimodal: Single text field (text_name)
- Tabular: No text fields (default)

#### `get_field_defaults(environ_vars: Dict[str, str]) -> Dict[str, str]`
**Purpose**: Load field default values from environment with 3-tier priority

**Algorithm**:
```python
1. Initialize field_defaults = {}
2. Load from FIELD_DEFAULTS JSON (base configuration):
   IF ENV_FIELD_DEFAULTS in environ_vars:
      Try parse JSON string to dictionary
      Store in field_defaults
3. Load from SPECIAL_FIELD_* prefix (overrides):
   FOR each (env_var, env_value) in environ_vars:
      IF env_var starts with "SPECIAL_FIELD_":
         field_name = env_var[len("SPECIAL_FIELD_"):].lower()
         field_defaults[field_name] = env_value (OVERRIDES JSON)
4. RETURN field_defaults
```

**Returns**: `Dict[str, str]` - Field name to default value mapping

**Priority (Highest to Lowest)**:
1. SPECIAL_FIELD_* (per-field override)
2. FIELD_DEFAULTS (JSON batch config)
3. Empty dict (use intelligent/generic defaults)

#### `generate_text_sample(field_name: str, field_defaults: Dict[str, str], default_text_value: str) -> str`
**Purpose**: Generate contextually appropriate text for a field using 3-tier priority

**Algorithm**:
```python
1. Priority 1 - User-provided (exact match):
   IF field_name in field_defaults:
      value = field_defaults[field_name]
      Try format with {timestamp} placeholder
      RETURN formatted or raw value
2. Priority 1 - User-provided (case-insensitive):
   field_lower = field_name.lower()
   FOR each (key, value) in field_defaults:
      IF key.lower() == field_lower:
         Try format with {timestamp}
         RETURN formatted or raw value
3. Priority 2 - Intelligent pattern-based:
   IF "chat" or "dialogue" or "conversation" in field_lower:
      RETURN "Hello, I need help with my order. Can you assist me?"
   ELIF "shiptrack" or "event" or "tracking" in field_lower:
      RETURN "Package shipped|In transit|Delivered"
   ELIF "description" or "desc" in field_lower:
      RETURN "Product description text for testing purposes"
   ELIF "comment" or "note" in field_lower:
      RETURN "Additional notes and comments for testing"
   ELIF "title" or "subject" in field_lower:
      RETURN "Sample title for testing"
   ELIF "message" or "msg" in field_lower:
      RETURN "Sample message content for testing"
4. Priority 3 - Generic fallback:
   RETURN default_text_value
```

**Returns**: `str` - Generated text sample

**Template Support**:
- `{timestamp}`: Replaced with "YYYY-MM-DD HH:MM:SS" format

### Source of Truth Component

#### `get_required_fields_from_model(model_dir: Path, hyperparams: Dict, var_type_list: List[List[str]]) -> Dict[str, Any]`
**Purpose**: Get required fields using SAME logic as inference handlers (critical for validation)

**Algorithm**:
```python
1. Initialize required = {
      "tabular_fields": [],
      "id_field": None,
      "text_fields": {},
      "model_type": "tabular",
      "field_order": [],
      "source": None
   }
2. Try feature_columns.txt (XGBoost/LightGBM priority):
   feature_file = model_dir / "feature_columns.txt"
   IF NOT feature_file.exists():
      Try extract from model.tar.gz if exists
   IF feature_file.exists():
      required["source"] = "feature_columns.txt"
      Parse file: read lines, extract column names
      required["tabular_fields"] = parsed_columns
      required["field_order"] = tabular_fields.copy()
      RETURN required
3. Fall back to hyperparameters (PyTorch models):
   required["source"] = "hyperparameters.json"
   model_type = detect_model_type(hyperparams)
   required["model_type"] = model_type
   Build field_order = []
   IF id_name in hyperparams:
      required["id_field"] = id_name
      Append to field_order
   IF model_type == "bimodal":
      Extract text_name
      required["text_fields"]["text_name"] = text_name
      Append to field_order
   ELIF model_type == "trimodal":
      Extract primary_text_name, secondary_text_name
      required["text_fields"]["primary_text_name"] = primary_text
      required["text_fields"]["secondary_text_name"] = secondary_text
      Append both to field_order
   FOR each (field_name, _) in var_type_list:
      required["tabular_fields"].append(field_name)
      Append to field_order
   required["field_order"] = field_order
4. RETURN required
```

**Returns**: `Dict[str, Any]` - Complete field requirements and ordering

**Critical for**:
- CSV field ordering (XGBoost/LightGBM)
- Payload validation
- Custom payload verification

### Validation Component

#### `validate_payload_completeness(payload: Dict, hyperparams: Dict, var_type_list: List[List[str]], model_dir: Optional[Path]) -> Tuple[bool, List[str]]`
**Purpose**: Validate payload contains all required fields for model type

**Algorithm**:
```python
1. Initialize required_fields = set()
2. IF model_dir provided and exists:
   a. Get required = get_required_fields_from_model(...)
   b. Add id_field if present
   c. Add all text_fields values
   d. Add all tabular_fields
3. ELSE (fallback to hyperparameters only):
   a. model_type = detect_model_type(hyperparams)
   b. Add id_name if in hyperparams
   c. IF bimodal: add text_name
   d. IF trimodal: add primary_text_name, secondary_text_name
   e. Add all fields from var_type_list
4. Validate:
   payload_fields = set(payload.keys())
   missing = required_fields - payload_fields
   extra = payload_fields - required_fields
5. Log warnings/info for missing/extra
6. RETURN (len(missing) == 0, list(missing))
```

**Returns**: `Tuple[bool, List[str]]` - (is_valid, missing_fields)

**Validation Checks**:
- All required fields present (ID, text fields, tabular fields)
- Reports missing fields with clear error
- Logs extra fields as info (not an error)

#### `log_payload_field_mapping(payload: Dict, hyperparams: Dict, var_type_list: List[List[str]]) -> None`
**Purpose**: Log comprehensive field mapping for debugging

**Algorithm**:
```python
1. Log header "=== PAYLOAD FIELD MAPPING ==="
2. Detect and log model_type
3. Log ID field if present
4. Log text fields based on model type:
   IF tabular: log "No text fields"
   IF bimodal: log text_name with preview (truncate to 50 chars)
   IF trimodal: log primary_text_name and secondary_text_name with previews
5. Log tabular fields count
6. FOR each (field_name, field_type) in var_type_list:
   Log field_name (field_type) = value
7. Log footer "=" * 40
```

**Output Example**:
```
=== PAYLOAD FIELD MAPPING ===
Model type: trimodal
  ID field: id = TEST_ID_20251129_211500
  Primary text field: chat_history = Hello, I need help with my order. Can you assi...
  Secondary text field: shiptrack_events = Package shipped|In transit|Delivered
  Tabular fields: 3 fields
    feature1 (NUMERIC) = 0.0
    feature2 (NUMERIC) = 0.0
    category1 (TEXT) = DEFAULT_TEXT
========================================
```

### Custom Payload Component

#### `load_custom_payload(custom_path: Path, content_type: str) -> Optional[Dict]`
**Purpose**: Load user-provided custom payload from file or directory

**Algorithm**:
```python
1. IF NOT custom_path.exists():
   Log warning, RETURN None
2. IF custom_path.is_dir():
   a. Search for JSON files (highest priority)
      IF found: load first, RETURN as dict
   b. Search for CSV files (second priority)
      IF found: load first row with pandas, RETURN as dict
   c. Search for Parquet files (third priority)
      IF found: load first row with pandas, RETURN as dict
   d. IF none found: log warning, RETURN None
3. ELIF custom_path.is_file():
   a. IF suffix == ".json":
      Load JSON, RETURN as dict
   b. ELIF suffix == ".csv":
      Load first row with pandas, RETURN as dict
   c. ELIF suffix == ".parquet":
      Load first row with pandas, RETURN as dict
   d. ELSE:
      Log warning about unsupported extension, RETURN None
4. Handle all exceptions: log error with traceback, RETURN None
```

**Returns**: `Optional[Dict]` - Loaded payload or None

**Supported Formats**:
- JSON (highest priority in directories)
- CSV (second priority, requires pandas)
- Parquet (third priority, requires pandas)

**Error Handling**:
- Logs warnings for missing files
- Handles pandas ImportError gracefully
- Logs all exceptions with tracebacks

### Hyperparameter Extraction Component

#### `extract_hyperparameters_from_tarball(input_model_dir: Path, working_directory: Path) -> Dict`
**Purpose**: Robustly extract hyperparameters.json from model artifacts using multiple fallback strategies

**Algorithm**:
```python
1. Set model_path = input_model_dir / "model.tar.gz"
2. Try Location 1: model.tar.gz as tarfile
   a. IF model_path exists and is_file:
      - Open as tar.gz
      - Search for hyperparameters.json member
      - Extract to working_directory if found
3. Try Location 2: model.tar.gz as directory
   a. IF model_path exists and is_dir:
      - Check for hyperparameters.json inside directory
4. Try Location 3: Direct file in model directory
   a. Check input_model_dir / "hyperparameters.json"
5. Try Location 4: Recursive search
   a. Use rglob("hyperparameters.json")
   b. Use first match found
6. IF still not found: raise FileNotFoundError with directory listing
7. Load JSON from found path
8. Copy to working_directory if not already there
9. RETURN parsed hyperparameters dictionary
```

**Parameters**:
- `input_model_dir` (Path): Model artifacts directory
- `working_directory` (Path): Working directory for extraction

**Returns**: `Dict` - Parsed hyperparameters dictionary

**Error Handling**: 
- Logs warning on tar processing errors, continues to other methods
- Lists directory contents if file not found in any location
- Raises FileNotFoundError if hyperparameters.json not found

**Robustness Features**:
- 4 fallback strategies ensure discovery in various packaging formats
- Handles both tar.gz files and incorrectly named directories
- Graceful degradation through fallback chain

## References

### Related Script Documentation

**Training Scripts** (generate hyperparameters consumed by this script):
- [`xgboost_training.py`](xgboost_training_script.md): XGBoost training - generates tabular model hyperparameters
- [`lightgbm_training.py`](lightgbm_training_script.md): LightGBM training - generates tabular model hyperparameters
- [`lightgbmmt_training.py`](lightgbmmt_training_script.md): LightGBM multi-task training
- [`pytorch_training.py`](pytorch_training_script.md): PyTorch training - generates multi-modal model hyperparameters

**Packaging Scripts** (create model artifacts with hyperparameters):
- [`package.py`](package_script.md): Creates packaged model.tar.gz with hyperparameters.json

**Inference Scripts** (use same field ordering logic):
- [`pytorch_inference_handler.py`](pytorch_inference_handler_script.md): PyTorch inference - uses same field ordering for multi-modal models
- [`xgboost_inference_handler.py`](xgboost_inference_handler_script.md): XGBoost inference - uses feature_columns.txt for field ordering
- [`pytorch_model_inference.py`](pytorch_model_inference_script.md): PyTorch model inference processing
- [`xgboost_model_inference.py`](xgboost_model_inference_script.md): XGBoost model inference processing

**Evaluation Scripts** (related testing workflows):
- [`pytorch_model_eval.py`](pytorch_model_eval_script.md): PyTorch model evaluation
- [`xgboost_model_eval.py`](xgboost_model_eval_script.md): XGBoost model evaluation
- [`lightgbm_model_eval.py`](lightgbm_model_eval_script.md): LightGBM model evaluation

**Calibration Scripts** (post-training processing):
- [`model_calibration.py`](model_calibration_script.md): Model calibration for probability scores

**Other Related Scripts**:
- [`mims_registration.py`](mims_registration_script.md): MIMS model registration workflow

### Related Source Code

- **Step Builder**: `src/cursus/steps/builders/builder_payload_step.py`
- **Config Class**: `src/cursus/steps/configs/config_payload_step.py`
- **Contract**: `src/cursus/steps/contracts/payload_contract.py`
- **Spec**: `src/cursus/steps/specs/payload_spec.py`
- **Design Doc**: `slipbox/2_project_planning/2025-11-29_payload_step_multimodal_expansion_implementation_plan.md`

### External References

- [SageMaker Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-hosting.html): AWS SageMaker model hosting and inference documentation
- [JSON Format Specification](https://www.json.org/): Official JSON data format specification
- [CSV Format Specification (RFC 4180)](https://tools.ietf.org/html/rfc4180): CSV file format specification
- [Parquet Format](https://parquet.apache.org/): Apache Parquet columnar storage format

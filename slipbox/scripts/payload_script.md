---
tags:
  - code
  - processing_script
  - payload_generation
  - mims
  - testing
keywords:
  - MIMS payload
  - inference testing
  - payload generation
  - sample data
  - model deployment
  - hyperparameters
  - content types
topics:
  - model deployment
  - testing workflows
  - MIMS integration
language: python
date of note: 2025-11-18
---

# Payload Script Documentation

## Overview

The `payload.py` script implements the MIMS (Model Inference Management System) payload generation workflow that creates sample inference payloads for testing deployed models. The script extracts field information from model hyperparameters and generates properly formatted test payloads in multiple content types (JSON, CSV) with configurable default values.

This script serves as a critical component in the model deployment testing pipeline, enabling automated generation of sample inference requests that match the exact input schema expected by the deployed model. It intelligently handles both numeric and text fields, supports custom field values through environment variables, and produces archived payloads ready for inference endpoint testing.

The payload generation process is driven by hyperparameters embedded in the model artifacts, ensuring that generated payloads always match the model's expected input format. This eliminates manual payload creation and reduces deployment testing errors.

## Purpose and Major Tasks

### Primary Purpose
Generate sample inference payloads from model hyperparameters in multiple content formats (JSON, CSV) for testing MIMS-deployed model endpoints with properly structured input data.

### Major Tasks

1. **Hyperparameter Extraction**: Extract and parse hyperparameters.json from model artifacts (tar.gz or directory) with multiple fallback strategies for robust discovery

2. **Variable List Creation**: Create typed variable lists from field information distinguishing numeric (tabular) and text (categorical) fields while excluding label and ID columns

3. **Environment Configuration**: Parse environment variables for content types, default values, and special field configurations with validation and error handling

4. **Multi-Format Payload Generation**: Generate payloads in multiple content types (application/json, text/csv) with proper field ordering and type-appropriate default values

5. **Special Field Handling**: Support custom field values through SPECIAL_FIELD_* environment variables with template substitution including timestamp formatting

6. **Payload File Creation**: Save generated payloads to files with descriptive names and proper formatting for each content type

7. **Archive Creation**: Create compressed tar.gz archive containing all payload samples with compression statistics and verification

8. **Logging and Verification**: Provide comprehensive logging including payload samples, file statistics, and archive creation details for debugging and validation

## Script Contract

### Entry Point
```
payload.py
```

### Input Paths

| Path | Location | Description |
|------|----------|-------------|
| `model_input` | `/opt/ml/processing/input/model` | Model artifacts directory containing hyperparameters.json (may be in tar.gz or loose) |

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
| `DEFAULT_TEXT_VALUE` | `"DEFAULT_TEXT"` | Default value for text/categorical fields in generated payloads |
| `SPECIAL_FIELD_<fieldname>` | None | Custom value for specific field (e.g., SPECIAL_FIELD_timestamp="{timestamp}") - supports {timestamp} placeholder |

#### Working Directory Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKING_DIRECTORY` | `"/tmp/mims_payload_work"` | Temporary directory for payload generation and intermediate files |

### Job Arguments

No command-line arguments are required. The script uses the standardized contract-based path structure.

## Input Data Structure

### Model Input Structure

The script accepts hyperparameters in multiple locations (tried in order):

**Location 1: Standard tar.gz file**
```
/opt/ml/processing/input/model/
└── model.tar.gz
    └── hyperparameters.json
```

**Location 2: Directory named model.tar.gz**
```
/opt/ml/processing/input/model/
└── model.tar.gz/              # Actually a directory
    └── hyperparameters.json
```

**Location 3: Direct file in model directory**
```
/opt/ml/processing/input/model/
└── hyperparameters.json
```

**Location 4: Recursive search**
```
/opt/ml/processing/input/model/
└── subdirectory/
    └── hyperparameters.json
```

### Hyperparameters Structure

```json
{
  "full_field_list": ["id", "feature1", "feature2", "category1", "label"],
  "tab_field_list": ["feature1", "feature2"],
  "cat_field_list": ["category1"],
  "label_name": "label",
  "id_name": "id",
  "pipeline_name": "my_pipeline",
  "pipeline_version": "1.0.0",
  "model_objective": "binary_classification"
}
```

**Field Definitions:**
- `full_field_list`: Complete list of all fields in the dataset
- `tab_field_list`: List of numeric/tabular fields
- `cat_field_list`: List of categorical/text fields
- `label_name`: Name of the label column (excluded from payloads)
- `id_name`: Name of the ID column (excluded from payloads)

**Note**: Only fields in `tab_field_list` or `cat_field_list` are included in payloads. The `full_field_list` may contain additional fields that are excluded.

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

**JSON Format (application/json)**
```json
{
  "feature1": "0.0",
  "feature2": "0.0",
  "category1": "DEFAULT_TEXT"
}
```

**CSV Format (text/csv)**
```csv
0.0,0.0,DEFAULT_TEXT
```

**Note**: CSV format contains only values (no header) in the same order as the variable list.

### Special Field Example

With environment variable `SPECIAL_FIELD_timestamp="{timestamp}"`:

```json
{
  "feature1": "0.0",
  "timestamp": "2025-11-18 20:45:30",
  "category1": "DEFAULT_TEXT"
}
```

## Key Functions and Tasks

### Hyperparameter Extraction Component

#### `extract_hyperparameters_from_tarball(input_model_dir: Path, working_directory: Path) -> Dict`
**Purpose**: Robustly extract hyperparameters.json from model artifacts using multiple fallback strategies

**Algorithm**:
```python
1. Set model_path = input_model_dir / "model.tar.gz"
2. Try Location 1: model.tar.gz as tarfile
   a. If model_path exists and is_file:
      - Open as tar.gz
      - Search for hyperparameters.json member
      - Extract to working_directory if found
3. Try Location 2: model.tar.gz as directory
   a. If model_path exists and is_dir:
      - Check for hyperparameters.json inside directory
4. Try Location 3: Direct file in model directory
   a. Check input_model_dir / "hyperparameters.json"
5. Try Location 4: Recursive search
   a. Use rglob("hyperparameters.json")
   b. Use first match found
6. If still not found: raise FileNotFoundError
7. Load JSON from found path
8. Copy to working_directory if not already there
9. Return parsed hyperparameters dictionary
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

#### `create_model_variable_list(full_field_list: List[str], tab_field_list: List[str], cat_field_list: List[str], label_name: str, id_name: str) -> List[List[str]]`
**Purpose**: Create typed variable list from field information for payload generation

**Algorithm**:
```python
1. Initialize model_var_list = []
2. For each field in full_field_list:
   a. If field == label_name or field == id_name:
      - Skip (continue to next field)
   b. Determine field_type:
      IF field in tab_field_list:
         field_type = "NUMERIC"
      ELIF field in cat_field_list:
         field_type = "TEXT"
      ELSE:
         field_type = "TEXT" (default)
   c. Append [field, field_type] to model_var_list
3. Return model_var_list
```

**Parameters**:
- `full_field_list` (List[str]): Complete list of field names
- `tab_field_list` (List[str]): Numeric field names
- `cat_field_list` (List[str]): Categorical field names
- `label_name` (str): Label column name to exclude
- `id_name` (str): ID column name to exclude

**Returns**: `List[List[str]]` - List of [field_name, field_type] pairs

**Complexity**:
- Time: O(n * m) where n = len(full_field_list), m = max(len(tab_field_list), len(cat_field_list))
- Space: O(n) for output list

**Type Determination Logic**:
- Priority 1: Check tab_field_list → NUMERIC
- Priority 2: Check cat_field_list → TEXT
- Default: TEXT (for unspecified fields)

### Environment Variable Parsing Component

#### `get_environment_content_types(environ_vars: Dict[str, str]) -> List[str]`
**Purpose**: Parse and return list of content types from environment variables

**Algorithm**:
```python
1. Get CONTENT_TYPES from environ_vars (default: "application/json")
2. Split by comma
3. Strip whitespace from each content type
4. Return list
```

**Example Input/Output**:
- Input: `"application/json, text/csv"`
- Output: `["application/json", "text/csv"]`

#### `get_environment_default_numeric_value(environ_vars: Dict[str, str]) -> float`
**Purpose**: Parse and return default numeric value with validation

**Algorithm**:
```python
1. Get DEFAULT_NUMERIC_VALUE from environ_vars (default: "0.0")
2. Try to convert to float
3. If ValueError:
   a. Log warning
   b. Return 0.0 (fallback)
4. Return converted float value
```

**Error Handling**: Catches ValueError, logs warning, returns safe default

#### `get_environment_default_text_value(environ_vars: Dict[str, str]) -> str`
**Purpose**: Get default text value from environment variables

**Algorithm**: Simple dictionary get with default "DEFAULT_TEXT"

#### `get_environment_special_fields(environ_vars: Dict[str, str]) -> Dict[str, str]`
**Purpose**: Extract special field configurations from environment variables

**Algorithm**:
```python
1. Initialize special_fields = {}
2. For each (env_var, env_value) in environ_vars:
   a. If env_var starts with "SPECIAL_FIELD_":
      - Extract field_name = env_var[len("SPECIAL_FIELD_"):].lower()
      - Store special_fields[field_name] = env_value
3. Return special_fields
```

**Example**:
- Input: `{"SPECIAL_FIELD_timestamp": "{timestamp}", "SPECIAL_FIELD_user_id": "test_user"}`
- Output: `{"timestamp": "{timestamp}", "user_id": "test_user"}`

### Default Value Resolution Component

#### `get_field_default_value(field_name: str, var_type: str, default_numeric_value: float, default_text_value: str, special_field_values: Dict[str, str]) -> str`
**Purpose**: Determine default value for a field based on type and special configurations

**Algorithm**:
```python
1. If var_type is TEXT or VariableType.TEXT:
   a. If field_name in special_field_values:
      - Get template from special_field_values[field_name]
      - Try format with timestamp=current datetime
      - If KeyError: raise ValueError with invalid placeholder message
      - Return formatted value
   b. Else:
      - Return default_text_value
2. Elif var_type is NUMERIC or VariableType.NUMERIC:
   a. Convert default_numeric_value to string
   b. Return string value
3. Else:
   a. Raise ValueError for unknown variable type
```

**Parameters**:
- `field_name` (str): Name of the field
- `var_type` (str): Field type (NUMERIC or TEXT)
- `default_numeric_value` (float): Default for numeric fields
- `default_text_value` (str): Default for text fields
- `special_field_values` (Dict[str, str]): Custom field values

**Returns**: `str` - Default value as string

**Error Handling**:
- Validates template placeholders
- Raises ValueError for invalid placeholders or unknown types

**Template Support**:
- `{timestamp}`: Replaced with current datetime in format "YYYY-MM-DD HH:MM:SS"

### Payload Generation Component

#### `generate_csv_payload(input_vars, default_numeric_value: float, default_text_value: str, special_field_values: Dict[str, str]) -> str`
**Purpose**: Generate CSV format payload (comma-separated values without header)

**Algorithm**:
```python
1. Initialize values = []
2. If input_vars is dict:
   For each (field_name, var_type) in input_vars.items():
      - Get default value using get_field_default_value()
      - Append to values
3. Else (list format):
   For each [field_name, var_type] in input_vars:
      - Get default value using get_field_default_value()
      - Append to values
4. Join values with comma separator
5. Return CSV string
```

**Returns**: `str` - Comma-separated values (no header)

**Example Output**: `"0.0,0.0,DEFAULT_TEXT"`

**Complexity**: O(n) where n = number of input variables

#### `generate_json_payload(input_vars, default_numeric_value: float, default_text_value: str, special_field_values: Dict[str, str]) -> str`
**Purpose**: Generate JSON format payload with field names as keys

**Algorithm**:
```python
1. Initialize payload = {}
2. If input_vars is dict:
   For each (field_name, var_type) in input_vars.items():
      - Get default value using get_field_default_value()
      - Set payload[field_name] = default_value
3. Else (list format):
   For each [field_name, var_type] in input_vars:
      - Get default value using get_field_default_value()
      - Set payload[field_name] = default_value
4. Convert payload dict to JSON string using json.dumps()
5. Return JSON string
```

**Returns**: `str` - JSON string with field name-value pairs

**Example Output**: `'{"feature1": "0.0", "feature2": "0.0", "category1": "DEFAULT_TEXT"}'`

**Complexity**: O(n) where n = number of input variables

#### `generate_sample_payloads(input_vars, content_types: List[str], default_numeric_value: float, default_text_value: str, special_field_values: Dict[str, str]) -> List[Dict[str, Union[str, dict]]]`
**Purpose**: Generate payloads for all specified content types

**Algorithm**:
```python
1. Initialize payloads = []
2. For each content_type in content_types:
   a. Initialize payload_info = {
         "content_type": content_type,
         "payload": None
      }
   b. If content_type == "text/csv":
      - Call generate_csv_payload()
      - Set payload_info["payload"]
   c. Elif content_type == "application/json":
      - Call generate_json_payload()
      - Set payload_info["payload"]
   d. Else:
      - Raise ValueError for unsupported content type
   e. Append payload_info to payloads
3. Return payloads list
```

**Returns**: `List[Dict]` - List of payload info dictionaries

**Supported Content Types**:
- `"text/csv"`: CSV format (comma-separated values)
- `"application/json"`: JSON format (field name-value pairs)

**Error Handling**: Raises ValueError for unsupported content types

### File Operations Component

#### `save_payloads(output_dir: str, input_vars, content_types: List[str], default_numeric_value: float, default_text_value: str, special_field_values: Dict[str, str]) -> List[str]`
**Purpose**: Save generated payloads to files with logging

**Algorithm**:
```python
1. Convert output_dir to Path object
2. Create output_dir (parents=True, exist_ok=True)
3. Initialize file_paths = []
4. Generate payloads using generate_sample_payloads()
5. Log header "===== GENERATED PAYLOAD SAMPLES ====="
6. For each payload_info in payloads:
   a. Extract content_type and payload
   b. Determine file extension:
      - ".csv" if content_type == "text/csv"
      - ".json" otherwise
   c. Create file_name = f"payload_{content_type.replace('/', '_')}_{i}{ext}"
   d. Construct file_path = output_dir / file_name
   e. Log content_type and payload sample
   f. Write payload to file
   g. Append file_path to file_paths
   h. Log file creation
7. Log footer "==================================="
8. Return file_paths
```

**Returns**: `List[str]` - List of created file paths

**File Naming Convention**:
- Format: `payload_{content_type_normalized}_{index}{extension}`
- Example: `payload_application_json_0.json`
- Example: `payload_text_csv_0.csv`

**Logging**:
- Logs each payload sample for debugging
- Logs file creation paths
- Provides clear section headers

**Complexity**: O(n * m) where n = number of content types, m = number of variables

#### `create_payload_archive(payload_files: List[str], output_dir: Path) -> str`
**Purpose**: Create compressed tar.gz archive of payload files with statistics

**Algorithm**:
```python
1. Set archive_path = output_dir / "payload.tar.gz"
2. Ensure output_dir exists
3. Log archive creation start
4. Initialize counters (total_size=0, files_added=0)
5. Open tar file in write-gzip mode
6. For each file_path in payload_files:
   a. Get base filename
   b. Calculate file size in MB
   c. Update total_size and files_added
   d. Log file addition
   e. Add file to tar with basename as arcname
7. Close tar file
8. Log summary statistics:
   - Files added count
   - Total uncompressed size
9. Verify archive exists and is file
10. Calculate compressed size
11. Calculate and log compression ratio
12. Return archive_path as string
```

**Parameters**:
- `payload_files` (List[str]): List of payload file paths to archive
- `output_dir` (Path): Output directory for archive

**Returns**: `str` - Path to created archive

**Error Handling**: 
- Catches all exceptions during tar creation
- Logs errors with traceback
- Re-raises exception

**Statistics Logged**:
- Files added count
- Total uncompressed size (MB)
- Compressed tar size (MB)
- Compression ratio (percentage)

**Complexity**: O(n * s) where n = number of files, s = average file size

### Main Orchestration Component

#### `main(input_paths: Dict[str, str], output_paths: Dict[str, str], environ_vars: Dict[str, str], job_args: Optional[argparse.Namespace]) -> str`
**Purpose**: Orchestrate complete payload generation workflow

**Algorithm**:
```python
1. Validate required paths in dictionaries
   - Check "model_input" in input_paths
   - Check "output_dir" in output_paths
2. Extract and convert paths to Path objects
3. Get working_directory from environ_vars or default
4. Set payload_sample_dir = working_directory / "payload_sample"
5. Log all configured paths
6. Extract hyperparameters from model artifacts
7. Extract field information from hyperparameters:
   - full_field_list, tab_field_list, cat_field_list
   - label_name, id_name
8. Create variable list:
   - Combine tab_field_list + cat_field_list
   - Call create_model_variable_list()
9. Parse environment configuration:
   - content_types, default_numeric_value
   - default_text_value, special_field_values
10. Extract metadata from hyperparameters:
    - pipeline_name, pipeline_version, model_objective
11. Ensure all directories exist:
    - working_directory, output_dir, payload_sample_dir
12. Generate and save payloads to payload_sample_dir
13. Create tar.gz archive from payload files
14. Log summary information:
    - Number of samples generated
    - Content types used
    - File locations
    - Input field information
15. Return archive_path
```

**Parameters**:
- `input_paths` (Dict[str, str]): Input path mappings
- `output_paths` (Dict[str, str]): Output path mappings
- `environ_vars` (Dict[str, str]): Environment variables
- `job_args` (Optional[argparse.Namespace]): Command-line arguments (unused)

**Returns**: `str` - Path to created payload.tar.gz file

**Error Handling**: 
- Top-level exception handler logs error and traceback
- Re-raises exception for step failure
- Validates required paths upfront

**Metadata Extraction**:
- Extracts pipeline_name, pipeline_version, model_objective from hyperparameters
- Uses defaults if not present

## Algorithms and Data Structures

### Multi-Location Hyperparameter Discovery Algorithm

**Problem**: Model artifacts may be packaged in various formats (tar.gz file, directory, loose files), requiring robust discovery mechanism

**Solution Strategy**:
1. Try most common format first (tar.gz file)
2. Fall back to alternate formats
3. Perform recursive search as last resort
4. Fail with descriptive error if not found

**Algorithm**:
```python
def extract_hyperparameters(input_dir):
    locations_to_try = [
        (is_tarfile, extract_from_tar),
        (is_directory_named_tar, read_from_directory),
        (is_direct_file, read_direct),
        (recursive_search, read_found)
    ]
    
    for check_func, read_func in locations_to_try:
        if check_func(input_dir):
            return read_func(input_dir)
    
    raise FileNotFoundError("Not found in any location")
```

**Complexity**:
- Time: O(n) worst case where n = total files (recursive search)
- Space: O(1) for checks, O(k) for file contents where k = hyperparameters size

**Key Features**:
- Ordered fallback strategy minimizes search time
- Early termination on first successful discovery
- Comprehensive error logging with directory contents

### Variable List Creation Algorithm

**Problem**: Convert unstructured field lists into typed variable list suitable for payload generation

**Solution Strategy**:
1. Filter out non-inference fields (label, ID)
2. Assign types based on membership in type-specific lists
3. Default to TEXT for unspecified fields

**Algorithm**:
```python
def create_variable_list(full_list, numeric_list, text_list, exclude_list):
    result = []
    for field in full_list:
        if field in exclude_list:
            continue
        
        if field in numeric_list:
            type = "NUMERIC"
        elif field in text_list:
            type = "TEXT"
        else:
            type = "TEXT"  # Default
        
        result.append([field, type])
    
    return result
```

**Complexity**:
- Time: O(n * m) where n = len(full_list), m = max(len(numeric_list), len(text_list))
- Space: O(n) for result list

**Type Determination Logic**:
- Membership check in numeric_list takes precedence
- Falls back to text_list check
- Defaults to TEXT for safety (more permissive)

### Multi-Format Payload Generation Algorithm

**Problem**: Generate payloads in different formats (JSON, CSV) from same variable list

**Solution Strategy**:
1. Abstract payload generation by content type
2. Use separate generator functions per format
3. Maintain consistent value ordering

**Algorithm**:
```python
def generate_payloads(variables, content_types, defaults):
    payloads = []
    
    for content_type in content_types:
        if content_type == "application/json":
            payload = generate_json(variables, defaults)
        elif content_type == "text/csv":
            payload = generate_csv(variables, defaults)
        else:
            raise ValueError(f"Unsupported: {content_type}")
        
        payloads.append({
            "content_type": content_type,
            "payload": payload
        })
    
    return payloads

def generate_json(variables, defaults):
    return json.dumps({
        name: get_default(name, type, defaults)
        for name, type in variables
    })

def generate_csv(variables, defaults):
    return ",".join([
        get_default(name, type, defaults)
        for name, type in variables
    ])
```

**Complexity**:
- Time: O(n * m) where n = number of content types, m = number of variables
- Space: O(n * m) for all payloads

**Key Features**:
- Format-agnostic variable list
- Consistent field ordering across formats
- Extensible to additional content types

## Performance Characteristics

### Complexity Analysis

| Operation | Time Complexity | Space Complexity | Notes |
|-----------|-----------------|------------------|-------|
| Hyperparameter extraction | O(n) | O(k) | n = files in recursive search, k = hyperparameters size |
| Variable list creation | O(n * m) | O(n) | n = fields, m = max type list size |
| CSV payload generation | O(n) | O(n) | n = number of variables |
| JSON payload generation | O(n) | O(n) | n = number of variables |
| Multi-format generation | O(n * m) | O(n * m) | n = content types, m = variables |
| File saving | O(n) | O(n) | n = number of payloads |
| Archive creation | O(n * s) | O(b) | n = files, s = avg size, b = buffer |
| Complete workflow | O(n * m + k) | O(n * m + k) | Dominated by payload generation |

### Processing Time Estimates

Based on typical model configurations:

| Model Complexity | Field Count | Content Types | Typical Time | Notes |
|------------------|-------------|---------------|--------------|-------|
| Simple | < 20 | 1 | < 1s | Single JSON payload |
| Standard | 20-100 | 2 | 1-2s | JSON + CSV payloads |
| Complex | 100-500 | 2 | 2-5s | Many fields, multiple types |
| Very Complex | > 500 | 3+ | 5-10s | High field count, many formats |

**Factors Affecting Performance**:
- Number of input variables
- Number of content types
- Special field template processing
- File system I/O speed
- Archive compression time

### Memory Usage

| Component | Memory Usage | Notes |
|-----------|--------------|-------|
| Hyperparameters | ~1-10KB | Small JSON file |
| Variable list | ~100B per field | Dominated by field names |
| Single payload | ~100B per field | String values |
| All payloads | ~100B * fields * types | Scales linearly |
| Archive buffer | ~1-10MB | Temporary during compression |
| Peak memory | ~10-50MB | For typical models |

**Memory Optimization**:
- Payloads stored as strings, not objects
- Archive created with streaming compression
- Temporary files cleaned up automatically

## Error Handling

### Error Types

#### Input Validation Errors

**Missing Required Path**
- **Cause**: `model_input` or `output_dir` not provided in input/output paths
- **Handling**: Raises ValueError with specific missing key message
- **Prevention**: Contract validation ensures paths are provided

**Hyperparameters Not Found**
- **Cause**: hyperparameters.json doesn't exist in any of the 4 checked locations
- **Handling**: Lists directory contents, raises FileNotFoundError
- **Impact**: Script cannot proceed without field information

**Invalid Hyperparameters Format**
- **Cause**: hyperparameters.json is not valid JSON or missing required keys
- **Handling**: JSON parsing error, exception raised with traceback
- **Impact**: Cannot extract field information for payload generation

#### Configuration Errors

**Invalid Default Numeric Value**
- **Cause**: DEFAULT_NUMERIC_VALUE environment variable is not a valid float
- **Handling**: Logs warning, falls back to 0.0
- **Impact**: Uses safe default value, continues processing

**Unsupported Content Type**
- **Cause**: Content type in CONTENT_TYPES is not "application/json" or "text/csv"
- **Handling**: Raises ValueError with unsupported content type message
- **Impact**: Prevents payload generation for that content type

**Invalid Template Placeholder**
- **Cause**: SPECIAL_FIELD_* template contains invalid placeholder (not {timestamp})
- **Handling**: Raises ValueError with specific placeholder error
- **Impact**: Cannot generate payload with special field

#### Processing Errors

**Directory Creation Failure**
- **Cause**: Permission issues, disk full, invalid path
- **Handling**: Logs error, ensure_directory() returns False
- **Impact**: May prevent payload saving or archive creation

**File Write Failure**
- **Cause**: Permission issues, disk full, I/O error
- **Handling**: Exception caught, logged with traceback, re-raised
- **Impact**: Payload file not created, archive may be incomplete

**Archive Creation Failure**
- **Cause**: Permission issues, disk full, file access issues
- **Handling**: Exception caught during tar creation, logged with traceback, re-raised
- **Impact**: No archive file created, operation fails

### Error Response Strategy

The script uses a layered error handling approach:

1. **Validation Layer**: Check required paths before processing
2. **Discovery Layer**: Multiple fallback strategies for hyperparameter discovery
3. **Operation Layer**: Try-catch around each major operation
4. **Logging Layer**: Comprehensive logging with tracebacks
5. **Graceful Degradation**: Use defaults when possible (numeric values, content types)

**Example Error Flow**:
```python
try:
    hyperparams = extract_hyperparameters(model_dir, working_dir)
except FileNotFoundError:
    logger.error("Hyperparameters not found in any location")
    list_directory_contents(model_dir)
    raise
```

## Best Practices

### For Production Deployments

1. **Validate Model Artifacts**
   - Verify hyperparameters.json exists in model.tar.gz
   - Check field lists are complete and accurate
   - Ensure field names match actual model inputs

2. **Configure Content Types**
   - Generate payloads for all content types your endpoint supports
   - Test with both JSON and CSV if endpoint accepts both
   - Verify content type matches endpoint configuration

3. **Use Special Fields Appropriately**
   - Configure SPECIAL_FIELD_* for fields requiring dynamic values
   - Use {timestamp} placeholder for time-sensitive fields
   - Avoid hardcoding test values that may cause confusion

4. **Test Generated Payloads**
   - Extract and inspect payload.tar.gz contents
   - Manually test payloads against staging endpoint
   - Verify field ordering in CSV matches model expectations

### For Development and Testing

1. **Verify Hyperparameter Extraction**
   - Check logs for successful hyperparameter discovery
   - Review extracted field lists
   - Confirm label and ID fields are excluded

2. **Test with Various Input Formats**
   - Test with standard model.tar.gz
   - Test with directory-based model artifacts
   - Test with loose hyperparameters.json
   - Verify all 4 discovery strategies work

3. **Customize Default Values**
   - Use realistic default values for numeric fields
   - Use meaningful text values for categorical fields
   - Test special field templates before deployment

4. **Review Generated Payloads**
   - Check payload samples in logs
   - Verify JSON structure matches model expectations
   - Ensure CSV field ordering is correct

### For Performance Optimization

1. **Minimize Content Types**
   - Generate only necessary content types
   - Use single content type for faster generation
   - Add additional types only when needed

2. **Optimize Field Lists**
   - Keep field lists minimal (remove unused fields)
   - Use efficient field naming (shorter names)
   - Ensure field lists don't include duplicates

3. **Configure Working Directory**
   - Use fast local storage for working directory
   - Avoid network file systems
   - Ensure sufficient disk space (3x payload size)

## Example Configurations

### Basic JSON Payload

```bash
# Standard configuration with JSON output only
# Uses all defaults

# Input:
# /opt/ml/processing/input/model/model.tar.gz
#   └── hyperparameters.json

# Output:
# /opt/ml/processing/output/payload.tar.gz
#   └── payload_application_json_0.json
```

**Use Case**: Simple model with JSON-only endpoint

### Multi-Format Payloads

```bash
export CONTENT_TYPES="application/json,text/csv"

# Generates both JSON and CSV payloads
# Useful for testing endpoints that support multiple formats
```

**Generated Files**:
- `payload_application_json_0.json`
- `payload_text_csv_0.csv`

**Use Case**: Model endpoint supporting multiple content types

### Custom Default Values

```bash
export DEFAULT_NUMERIC_VALUE="1.0"
export DEFAULT_TEXT_VALUE="SAMPLE_VALUE"

# Customizes default values for all fields
# Useful for more realistic test data
```

**Generated Payload**:
```json
{
  "numeric_feature": "1.0",
  "text_feature": "SAMPLE_VALUE"
}
```

**Use Case**: Testing with non-zero default values

### Special Field Templates

```bash
export SPECIAL_FIELD_timestamp="{timestamp}"
export SPECIAL_FIELD_user_id="test_user_123"
export SPECIAL_FIELD_session_id="session_abc"

# Configures specific fields with custom values
# {timestamp} placeholder replaced at runtime
```

**Generated Payload**:
```json
{
  "feature1": "0.0",
  "timestamp": "2025-11-18 20:45:30",
  "user_id": "test_user_123",
  "session_id": "session_abc",
  "category": "DEFAULT_TEXT"
}
```

**Use Case**: Models requiring specific field values like timestamps or identifiers

### Custom Working Directory

```bash
export WORKING_DIRECTORY="/mnt/fast-storage/payload_work"

# Uses custom working directory for better I/O performance
# Useful when /tmp has limited space or slow I/O
```

**Use Case**: Large model configurations or environments with constrained /tmp space

## Integration Patterns

### Upstream Integration

```
PackageStep
   ↓ (outputs: model.tar.gz with hyperparameters.json)
PayloadStep
   ↓ (outputs: payload.tar.gz)
```

**Key Connection**: Payload step reads hyperparameters embedded in packaged model

### Downstream Integration

```
PayloadStep
   ↓ (payload.tar.gz)
Manual Testing / Automated Testing
   ↓ (test payloads against endpoint)
Endpoint Validation
   ↓ (verify correct responses)
```

**Usage**: Payloads used for immediate endpoint testing after deployment

### Complete Workflow Example

```
1. XGBoostTraining
   ↓ model.tar.gz (with hyperparameters)
2. ModelCalibration (optional)
   ↓ calibration artifacts
3. Package
   ↓ packaged_model.tar.gz (contains hyperparameters.json)
4. Payload (THIS STEP)
   Inputs:
   - packaged_model.tar.gz from step 3
   - Environment configuration for payload generation
   Output:
   ↓ payload.tar.gz (test samples)
5. ModelRegistration (MIMS)
   ↓ registered model in MIMS
6. Deployment
   ↓ live endpoint
7. Testing
   Uses payloads from step 4 to test endpoint
```

**Key Integration Points**:
- **Package outputs**: Reads hyperparameters from packaged model
- **Configuration inputs**: Uses environment variables for customization
- **Testing usage**: Generated payloads used for endpoint validation

## Troubleshooting

### Hyperparameters Not Found

**Symptom**: "hyperparameters.json not found in model artifacts" error

**Common Causes**:
1. **Missing file**: hyperparameters.json not included in model.tar.gz
2. **Wrong location**: File in unexpected directory structure
3. **Incorrect packaging**: Model artifacts not properly packaged

**Solution**:
```bash
# Check model.tar.gz contents
tar -tzf /opt/ml/processing/input/model/model.tar.gz | grep hyperparameters

# Should see:
# hyperparameters.json

# Check logs for attempted locations
grep "Looking for hyperparameters" processing_logs.txt

# Manually verify file exists
find /opt/ml/processing/input/model -name "hyperparameters.json"
```

### Missing Field Information

**Symptom**: Payload has no fields or missing expected fields

**Common Causes**:
1. **Empty field lists**: tab_field_list and cat_field_list are empty
2. **All fields excluded**: All fields are label or ID
3. **Incorrect field lists**: Fields not properly categorized

**Solution**:
```bash
# Check hyperparameters content
cat /tmp/mims_payload_work/hyperparameters.json | jq '.tab_field_list, .cat_field_list'

# Review field lists in logs
grep "Input field information" processing_logs.txt

# Verify field categorization
# Ensure tab_field_list and cat_field_list contain actual fields
```

### Invalid Content Type

**Symptom**: "Unsupported content type" error

**Common Causes**:
1. **Typo in CONTENT_TYPES**: Misspelled content type
2. **Unsupported format**: Requested format not implemented
3. **Extra whitespace**: Leading/trailing spaces in content type

**Solution**:
```bash
# Check environment variable
echo $CONTENT_TYPES

# Valid values:
# - application/json
# - text/csv
# - application/json,text/csv

# Fix typos
export CONTENT_TYPES="application/json,text/csv"

# Remove extra spaces
export CONTENT_TYPES="application/json,text/csv"  # Not "application/json , text/csv"
```

### Template Formatting Error

**Symptom**: "Invalid placeholder in template" error

**Common Causes**:
1. **Invalid placeholder**: Used placeholder other than {timestamp}
2. **Typo in placeholder**: Misspelled {timestamp}
3. **Nested braces**: Invalid template syntax

**Solution**:
```bash
# Check special field configuration
env | grep SPECIAL_FIELD

# Valid template:
export SPECIAL_FIELD_timestamp="{timestamp}"

# Invalid templates:
# export SPECIAL_FIELD_custom="{custom_value}"  # Unsupported placeholder
# export SPECIAL_FIELD_time="{time stamp}"      # Space in placeholder
# export SPECIAL_FIELD_date="{{timestamp}}"     # Double braces

# Currently only {timestamp} placeholder is supported
```

### Archive Not Created

**Symptom**: payload.tar.gz file not found in output directory

**Common Causes**:
1. **Permission issues**: Cannot write to output directory
2. **Disk space**: Insufficient space for archive
3. **Previous error**: Earlier failure prevented archive creation

**Solution**:
```bash
# Check output directory permissions
ls -ld /opt/ml/processing/output

# Check disk space
df -h /opt/ml/processing/output

# Review logs for errors
grep -i "error\|exception" processing_logs.txt

# Check if payload files were created
ls -la /tmp/mims_payload_work/payload_sample/

# If files exist but archive missing, check archive creation logs
grep "Creating payload archive" processing_logs.txt
```

### Empty Payloads

**Symptom**: Generated payloads contain no fields

**Common Causes**:
1. **No inference fields**: All fields are label/ID/excluded
2. **Empty field lists**: tab_field_list and cat_field_list both empty
3. **All fields excluded**: Fields not in either list

**Solution**:
```bash
# Check what fields are available
cat /tmp/mims_payload_work/hyperparameters.json | jq '{
  full: .full_field_list,
  tab: .tab_field_list,
  cat: .cat_field_list,
  label: .label_name,
  id: .id_name
}'

# Verify actual vs expected
# Ensure tab_field_list + cat_field_list contains features
# Ensure these don't just contain label and ID

# Check variable list in logs
grep "Total fields:" processing_logs.txt
```

## References

### Related Scripts

- [`package.py`](package_script.md): Creates packaged model.tar.gz containing hyperparameters.json that this script consumes
- [`xgboost_training.py`](xgboost_training_script.md): Generates initial hyperparameters during training
- [`lightgbm_training.py`](lightgbm_training_script.md): Generates initial hyperparameters during training
- [`pytorch_training.py`](pytorch_training_script.md): Generates initial hyperparameters during training

### Related Documentation

- **Step Builder**: `src/cursus/steps/builders/builder_payload_step.py`
- **Config Class**: `src/cursus/steps/configs/config_payload_step.py`
- **Contract**: [`src/cursus/steps/contracts/payload_contract.py`](../../src/cursus/steps/contracts/payload_contract.py)

### External References

- [MIMS Documentation](https://w.amazon.com/bin/view/MIMS/): Internal MIMS deployment system documentation
- [SageMaker Inference](https://docs.aws.amazon.com/sagemaker/latest/dg/how-it-works-hosting.html): SageMaker model hosting and inference
- [JSON Format](https://www.json.org/): JSON data format specification
- [CSV Format](https://tools.ietf.org/html/rfc4180): CSV file format specification

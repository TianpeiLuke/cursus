---
tags:
  - code
  - validation
  - alignment
  - static_analysis
  - script_analyzer
keywords:
  - script analyzer
  - AST parsing
  - static code analysis
  - path references
  - environment variables
  - import statements
  - file operations
topics:
  - alignment validation
  - static analysis
  - script analysis
language: python
date of note: 2025-08-19
---

# Script Analyzer

## Overview

The `ScriptAnalyzer` class performs comprehensive static analysis of Python scripts using AST (Abstract Syntax Tree) parsing to extract usage patterns. This component identifies path references, environment variable access, import statements, argument parsing patterns, and file operations, providing the foundational data for script-contract alignment validation.

## Core Functionality

### Static Analysis Features

The Script Analyzer provides comprehensive script analysis capabilities:

1. **Path Reference Extraction**: Identifies hardcoded paths and path construction patterns
2. **Environment Variable Analysis**: Detects environment variable access patterns
3. **Import Statement Analysis**: Extracts module imports and framework detection
4. **Argument Definition Analysis**: Identifies command-line argument parsing
5. **File Operation Analysis**: Detects file I/O operations across multiple frameworks
6. **Step Type Detection**: Determines SageMaker step types and ML frameworks

### Key Components

#### ScriptAnalyzer Class

The main analyzer class that orchestrates all static analysis operations:

```python
class ScriptAnalyzer:
    """
    Analyzes Python script source code to extract usage patterns.
    
    Uses AST parsing to identify:
    - Path references and construction
    - Environment variable access
    - Import statements
    - Function definitions and calls
    - Argument parsing patterns
    """
```

## Core Analysis Methods

### Path Reference Analysis

#### extract_path_references()

Extracts all path references from the script using AST parsing:

**Detection Strategies**:
- **String Literal Analysis**: Identifies path-like strings in the code
- **Path Construction Detection**: Recognizes `os.path.join` and pathlib operations
- **Heuristic Path Identification**: Uses patterns to identify path-like strings

**Path Indicators**:
- SageMaker paths: `/opt/ml/`, `/tmp/`, `/var/`
- Relative paths: `./`, `../`
- File extensions and multiple path separators
- Windows path patterns with backslashes

**PathReference Objects**:
```python
PathReference(
    path="/opt/ml/input/data",
    line_number=42,
    context="data_path = '/opt/ml/input/data'",
    is_hardcoded=True,
    construction_method=None
)
```

**Construction Pattern Detection**:
- Identifies `os.path.join()` calls
- Extracts path components from construction
- Marks constructed paths as non-hardcoded
- Provides construction method information

### Environment Variable Analysis

#### extract_env_var_access()

Detects all environment variable access patterns:

**Supported Access Patterns**:
- `os.environ['VAR']` - Direct dictionary access
- `os.getenv('VAR', 'default')` - Function-based access with optional default
- `os.environ.get('VAR', 'default')` - Method-based access with optional default

**EnvVarAccess Objects**:
```python
EnvVarAccess(
    variable_name="SM_MODEL_DIR",
    line_number=15,
    context="model_dir = os.getenv('SM_MODEL_DIR', '/opt/ml/model')",
    access_method="os.getenv",
    has_default=True,
    default_value="/opt/ml/model"
)
```

**Default Value Detection**:
- Identifies when default values are provided
- Extracts default value literals
- Tracks access methods for validation purposes

### Import Statement Analysis

#### extract_imports()

Extracts all import statements from the script:

**Import Types Supported**:
- Standard imports: `import pandas as pd`
- From imports: `from sklearn.ensemble import RandomForestClassifier`
- Multiple imports: `from module import item1, item2`

**ImportStatement Objects**:
```python
ImportStatement(
    module_name="pandas",
    import_alias="pd",
    line_number=5,
    is_from_import=False,
    imported_items=[]
)
```

**Framework Detection Integration**:
- Provides import data for framework detection
- Supports step type classification
- Enables framework-specific validation

### Argument Definition Analysis

#### extract_argument_definitions()

Identifies command-line argument parsing patterns:

**Argument Parser Detection**:
- Recognizes `add_argument()` calls
- Extracts argument names, types, and properties
- Handles required/optional argument classification

**ArgumentDefinition Objects**:
```python
ArgumentDefinition(
    argument_name="input_path",
    line_number=25,
    is_required=True,
    has_default=False,
    default_value=None,
    argument_type="str",
    choices=None
)
```

**Property Extraction**:
- **Required Status**: Identifies required vs optional arguments
- **Default Values**: Extracts default values when provided
- **Type Information**: Captures argument type specifications
- **Choices**: Identifies valid argument choices when specified

### File Operation Analysis

#### extract_file_operations()

Comprehensive file operation detection across multiple frameworks:

**Standard Python I/O**:
- `open()` calls with mode detection
- File mode analysis (read/write/append)
- Context manager usage

**Framework-Specific Operations**:
- **Pandas**: `pd.read_csv()`, `df.to_csv()`
- **Pickle**: `pickle.load()`, `pickle.dump()`
- **JSON**: `json.load()`, `json.dump()`
- **Tarfile**: `tarfile.open()` with mode detection
- **Shutil**: `shutil.copy()`, `shutil.move()`
- **Pathlib**: `Path.read_text()`, `Path.write_text()`
- **XGBoost**: `model.load_model()`, `model.save_model()`
- **Matplotlib**: `plt.savefig()`

**FileOperation Objects**:
```python
FileOperation(
    file_path="/opt/ml/output/model.pkl",
    operation_type="write",
    line_number=78,
    context="pickle.dump(model, open('/opt/ml/output/model.pkl', 'wb'))",
    mode="wb",
    method="pickle.dump"
)
```

**Operation Type Classification**:
- **Read Operations**: File input, data loading
- **Write Operations**: File output, data saving
- **Directory Operations**: Directory creation, traversal

### Advanced Analysis Features

#### get_all_analysis_results()

Provides comprehensive analysis results with step type awareness:

**Enhanced Analysis**:
- Basic pattern extraction (paths, environment variables, imports, arguments, file operations)
- Step type detection using registry integration
- Framework detection from import analysis
- Step type-specific pattern detection

**Step Type Integration**:
- Uses `detect_step_type_from_registry()` for canonical step type identification
- Applies `detect_framework_from_imports()` for ML framework detection
- Includes training-specific pattern detection for training scripts

**Comprehensive Results Structure**:
```python
{
    'script_path': '/path/to/script.py',
    'path_references': [PathReference, ...],
    'env_var_accesses': [EnvVarAccess, ...],
    'imports': [ImportStatement, ...],
    'argument_definitions': [ArgumentDefinition, ...],
    'file_operations': [FileOperation, ...],
    'step_type': 'Training',
    'framework': 'xgboost',
    'step_type_patterns': {...}
}
```

### Utility Methods

#### has_main_function()

Detects presence of a main function in the script:

**Detection Logic**:
- Scans AST for function definitions named 'main'
- Returns boolean indicating presence
- Supports testability pattern validation

#### has_main_block()

Detects presence of `if __name__ == '__main__':` block:

**Detection Logic**:
- Identifies the standard Python main block pattern
- Supports entry point structure validation
- Enables testability pattern analysis

## AST Visitor Patterns

### Specialized Visitor Classes

The Script Analyzer uses specialized AST visitor classes for different analysis types:

#### PathVisitor
- **String Literal Analysis**: Examines all string literals for path patterns
- **Path Construction**: Detects `os.path.join` and pathlib operations
- **Heuristic Identification**: Uses multiple indicators to identify paths

#### EnvVarVisitor
- **Subscript Operations**: Detects `os.environ['VAR']` patterns
- **Function Calls**: Identifies `os.getenv()` and `os.environ.get()` calls
- **Default Value Extraction**: Captures default values when provided

#### ImportVisitor
- **Import Statements**: Processes both `import` and `from ... import` statements
- **Alias Handling**: Captures import aliases and renamed imports
- **Module Tracking**: Maintains complete import information

#### ArgumentVisitor
- **Argparse Detection**: Identifies `add_argument()` method calls
- **Property Extraction**: Captures argument properties from keyword arguments
- **Type Analysis**: Extracts type information and validation constraints

#### FileOpVisitor
- **Multi-Framework Support**: Handles operations from various Python libraries
- **Operation Classification**: Determines read vs write operations
- **Path Extraction**: Extracts file paths from various call patterns

## Integration Points

### Alignment Validation Framework
- **Script Analysis Models**: Uses data models from script_analysis_models
- **Validation Input**: Provides structured data for validation processes
- **Pattern Recognition**: Supports pattern-based validation rules

### Step Type Detection
- **Registry Integration**: Uses step registry for canonical step type detection
- **Framework Detection**: Integrates with framework detection utilities
- **Pattern Analysis**: Supports step type-specific pattern recognition

### Testability Validation
- **Structure Analysis**: Provides data for testability pattern validation
- **Main Function Detection**: Supports testability structure requirements
- **Entry Point Analysis**: Enables testability compliance checking

## Usage Patterns

### Basic Script Analysis

```python
analyzer = ScriptAnalyzer('/path/to/script.py')

# Extract specific patterns
path_refs = analyzer.extract_path_references()
env_vars = analyzer.extract_env_var_access()
imports = analyzer.extract_imports()
arguments = analyzer.extract_argument_definitions()
file_ops = analyzer.extract_file_operations()

# Print analysis results
for path_ref in path_refs:
    print(f"Path: {path_ref.path} (line {path_ref.line_number})")
```

### Comprehensive Analysis

```python
analyzer = ScriptAnalyzer('/path/to/script.py')

# Get complete analysis results
results = analyzer.get_all_analysis_results()

print(f"Script: {results['script_path']}")
print(f"Step Type: {results['step_type']}")
print(f"Framework: {results['framework']}")
print(f"Paths found: {len(results['path_references'])}")
print(f"Environment variables: {len(results['env_var_accesses'])}")
print(f"File operations: {len(results['file_operations'])}")
```

### Validation Integration

```python
def analyze_script_for_validation(script_path):
    """Analyze script for alignment validation."""
    try:
        analyzer = ScriptAnalyzer(script_path)
        return analyzer.get_all_analysis_results()
    except ValueError as e:
        return {'error': str(e), 'script_path': script_path}

# Use in validation pipeline
analysis_results = analyze_script_for_validation('training_script.py')
if 'error' not in analysis_results:
    # Proceed with validation using analysis results
    validation_issues = validate_script_contract_alignment(
        analysis_results, contract, script_name
    )
```

## Benefits

### Comprehensive Analysis
- **Multi-Pattern Detection**: Identifies various code patterns in single pass
- **Framework Awareness**: Recognizes patterns from multiple ML frameworks
- **Context Preservation**: Maintains code context for detailed reporting

### Robust Parsing
- **AST-Based Analysis**: Uses Python's built-in AST parsing for accuracy
- **Error Handling**: Graceful handling of syntax errors and parsing issues
- **Line Number Tracking**: Provides precise location information for all patterns

### Extensible Architecture
- **Visitor Pattern**: Easy addition of new analysis patterns
- **Modular Design**: Separate analysis methods for different pattern types
- **Integration Ready**: Designed for integration with validation frameworks

## Design Considerations

### Performance Optimization
- **Single AST Parse**: Parses script once and reuses AST for all analysis
- **Efficient Visitors**: Optimized AST traversal patterns
- **Lazy Evaluation**: Analysis methods called only when needed

### Accuracy and Reliability
- **Heuristic Validation**: Multiple indicators for pattern identification
- **Context Analysis**: Considers surrounding code for accurate detection
- **Error Resilience**: Continues analysis despite individual pattern failures

### Extensibility
- **Plugin Architecture**: Easy addition of new visitor classes
- **Pattern Extension**: Simple addition of new detection patterns
- **Framework Support**: Extensible framework-specific operation detection

## Future Enhancements

### Advanced Analysis
- **Control Flow Analysis**: Understanding of conditional logic and loops
- **Data Flow Analysis**: Tracking data flow through script execution
- **Semantic Analysis**: Enhanced understanding of code semantics

### Enhanced Detection
- **Machine Learning Patterns**: ML-specific pattern recognition
- **Custom Framework Support**: Support for custom and proprietary frameworks
- **Dynamic Analysis**: Runtime behavior analysis integration

### Integration Improvements
- **IDE Integration**: Real-time analysis in development environments
- **Incremental Analysis**: Efficient re-analysis of changed code sections
- **Parallel Processing**: Concurrent analysis of multiple scripts

## Conclusion

The Script Analyzer provides essential static analysis capabilities for the alignment validation framework, extracting comprehensive usage patterns from Python scripts through sophisticated AST parsing. By identifying paths, environment variables, imports, arguments, and file operations across multiple frameworks, it enables accurate script-contract alignment validation. This component is fundamental to understanding script behavior and ensuring proper alignment with contract specifications.

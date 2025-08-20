---
tags:
  - code
  - validation
  - alignment
  - static_analysis
  - script_analysis
keywords:
  - script analyzer
  - AST parsing
  - static analysis
  - path references
  - environment variables
  - argument parsing
  - file operations
topics:
  - validation framework
  - static code analysis
  - script pattern extraction
language: python
date of note: 2025-08-19
---

# Script Analyzer

## Overview

The `ScriptAnalyzer` class provides comprehensive static analysis of Python scripts using AST (Abstract Syntax Tree) parsing. It extracts usage patterns including path references, environment variable access, import statements, argument parsing patterns, and file operations to support alignment validation and script compliance checking.

## Architecture

### Core Capabilities

1. **AST-Based Analysis**: Deep parsing of Python source code using Abstract Syntax Trees
2. **Pattern Extraction**: Identification of various usage patterns and code constructs
3. **Multi-Dimensional Analysis**: Comprehensive extraction across multiple code aspects
4. **Step Type Awareness**: Enhanced analysis with step type and framework detection
5. **Context Preservation**: Maintains line numbers and code context for all findings

### Analysis Dimensions

The analyzer provides comprehensive analysis across multiple code dimensions:

- **Path Usage**: Hardcoded paths and dynamic path construction
- **Environment Access**: Environment variable access patterns
- **Import Analysis**: Module imports and dependencies
- **Argument Parsing**: Command-line argument definitions
- **File Operations**: File read/write operations across multiple libraries
- **Code Structure**: Main functions and execution blocks

## Implementation Details

### Class Structure

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

### Key Methods

#### `extract_path_references() -> List[PathReference]`

Extracts all path references from the script using AST traversal:

**Detection Patterns:**
- Hardcoded path strings: `/opt/ml/processing/input/data.csv`
- Dynamic path construction: `os.path.join()` calls
- Path-like string identification using heuristics
- Context preservation with line numbers

**PathReference Object:**
```python
PathReference(
    path: str,
    line_number: int,
    context: str,
    is_hardcoded: bool,
    construction_method: Optional[str]
)
```

#### `extract_env_var_access() -> List[EnvVarAccess]`

Identifies environment variable access patterns:

**Supported Patterns:**
- `os.environ['VAR']`: Direct dictionary access
- `os.getenv('VAR', 'default')`: Function call with optional default
- `os.environ.get('VAR', 'default')`: Method call with optional default

**EnvVarAccess Object:**
```python
EnvVarAccess(
    variable_name: str,
    line_number: int,
    context: str,
    access_method: str,
    has_default: bool,
    default_value: Optional[str]
)
```

#### `extract_imports() -> List[ImportStatement]`

Extracts all import statements from the script:

**Import Types:**
- Direct imports: `import pandas as pd`
- From imports: `from sklearn.ensemble import RandomForestClassifier`
- Module aliases and imported items tracking

#### `extract_argument_definitions() -> List[ArgumentDefinition]`

Identifies command-line argument definitions from argparse usage:

**Extracted Properties:**
- Argument names (normalized from `--arg-name` to `arg_name`)
- Required/optional status
- Default values
- Argument types
- Choice constraints

**ArgumentDefinition Object:**
```python
ArgumentDefinition(
    argument_name: str,
    line_number: int,
    is_required: bool,
    has_default: bool,
    default_value: Any,
    argument_type: Optional[str],
    choices: Optional[List[str]]
)
```

#### `extract_file_operations() -> List[FileOperation]`

Comprehensive extraction of file operations across multiple libraries:

**Supported Operations:**
- **Standard Library**: `open()`, `with open()`
- **Archive Operations**: `tarfile.open()`
- **File System**: `shutil.copy()`, `shutil.move()`
- **Path Operations**: `pathlib.Path` methods
- **Data Science**: `pandas.read_csv()`, `DataFrame.to_csv()`
- **Serialization**: `pickle.load()`, `json.dump()`
- **ML Models**: `model.save_model()`, `model.load_model()`
- **Visualization**: `plt.savefig()`
- **Directory Operations**: `Path.glob()`

**FileOperation Object:**
```python
FileOperation(
    file_path: str,
    operation_type: str,  # 'read', 'write', 'unknown'
    line_number: int,
    context: str,
    mode: Optional[str],
    method: str
)
```

#### `get_all_analysis_results() -> Dict[str, Any]`

Provides comprehensive analysis results with step type awareness:

```python
{
    'script_path': str,
    'path_references': List[PathReference],
    'env_var_accesses': List[EnvVarAccess],
    'imports': List[ImportStatement],
    'argument_definitions': List[ArgumentDefinition],
    'file_operations': List[FileOperation],
    'step_type': str,  # Detected step type
    'framework': str,  # Detected framework
    'step_type_patterns': Dict[str, Any]  # Step-specific patterns
}
```

#### `has_main_function() -> bool`

Checks for the presence of a `main()` function definition.

#### `has_main_block() -> bool`

Checks for the presence of a `if __name__ == '__main__':` block.

## Usage Examples

### Basic Script Analysis

```python
from cursus.validation.alignment.static_analysis.script_analyzer import ScriptAnalyzer

# Initialize analyzer with script path
script_path = "/path/to/processing_script.py"
analyzer = ScriptAnalyzer(script_path)

# Get comprehensive analysis
results = analyzer.get_all_analysis_results()

print(f"Script: {results['script_path']}")
print(f"Step type: {results['step_type']}")
print(f"Framework: {results['framework']}")
```

### Path Reference Analysis

```python
# Extract path references
path_refs = analyzer.extract_path_references()

for path_ref in path_refs:
    print(f"Line {path_ref.line_number}: {path_ref.path}")
    print(f"  Hardcoded: {path_ref.is_hardcoded}")
    if path_ref.construction_method:
        print(f"  Method: {path_ref.construction_method}")
    print(f"  Context: {path_ref.context[:50]}...")
```

### Environment Variable Analysis

```python
# Extract environment variable access
env_accesses = analyzer.extract_env_var_access()

for env_access in env_accesses:
    print(f"Line {env_access.line_number}: {env_access.variable_name}")
    print(f"  Method: {env_access.access_method}")
    print(f"  Has default: {env_access.has_default}")
    if env_access.default_value:
        print(f"  Default: {env_access.default_value}")
```

### Argument Definition Analysis

```python
# Extract argument definitions
arguments = analyzer.extract_argument_definitions()

for arg in arguments:
    print(f"Argument: {arg.argument_name}")
    print(f"  Required: {arg.is_required}")
    print(f"  Type: {arg.argument_type}")
    if arg.has_default:
        print(f"  Default: {arg.default_value}")
    if arg.choices:
        print(f"  Choices: {arg.choices}")
```

### File Operation Analysis

```python
# Extract file operations
file_ops = analyzer.extract_file_operations()

for file_op in file_ops:
    print(f"Line {file_op.line_number}: {file_op.operation_type}")
    print(f"  Path: {file_op.file_path}")
    print(f"  Method: {file_op.method}")
    if file_op.mode:
        print(f"  Mode: {file_op.mode}")
```

### Import Analysis

```python
# Extract imports
imports = analyzer.extract_imports()

for import_stmt in imports:
    print(f"Line {import_stmt.line_number}: {import_stmt.module_name}")
    if import_stmt.import_alias:
        print(f"  Alias: {import_stmt.import_alias}")
    if import_stmt.is_from_import:
        print(f"  Items: {import_stmt.imported_items}")
```

### Code Structure Analysis

```python
# Check code structure
has_main_func = analyzer.has_main_function()
has_main_block = analyzer.has_main_block()

print(f"Has main function: {has_main_func}")
print(f"Has main block: {has_main_block}")
```

## Integration Points

### Alignment Validation Framework

The ScriptAnalyzer integrates with the alignment validation system:

```python
class AlignmentValidator:
    def validate_script_patterns(self, script_path):
        analyzer = ScriptAnalyzer(script_path)
        results = analyzer.get_all_analysis_results()
        
        # Validate path usage
        path_issues = self.validate_path_references(results['path_references'])
        
        # Validate environment variable usage
        env_issues = self.validate_env_var_access(results['env_var_accesses'])
        
        # Validate argument definitions
        arg_issues = self.validate_arguments(results['argument_definitions'])
        
        return {
            'path_issues': path_issues,
            'env_issues': env_issues,
            'arg_issues': arg_issues,
            'step_type': results['step_type'],
            'framework': results['framework']
        }
```

### Static Analysis Pipeline

Works as the central component of static analysis:

- **AST Foundation**: Provides AST parsing foundation for other analyzers
- **Pattern Extraction**: Extracts patterns for specialized analyzers
- **Context Preservation**: Maintains code context for validation reporting
- **Integration Hub**: Coordinates with ImportAnalyzer and PathExtractor

### Contract Validation

Supports contract alignment validation:

- **Argument Validation**: Validates script arguments against contracts
- **Path Validation**: Ensures path usage matches contract specifications
- **Environment Validation**: Validates environment variable usage
- **Import Validation**: Checks import requirements against contracts

## Advanced Features

### AST Visitor Pattern

Sophisticated AST traversal using the visitor pattern:

- **Specialized Visitors**: Custom visitors for different analysis types
- **Context Preservation**: Maintains line numbers and code context
- **Error Resilience**: Graceful handling of AST parsing issues
- **Extensibility**: Easy addition of new pattern detection

### Multi-Library File Operation Detection

Comprehensive file operation detection across libraries:

- **Standard Library**: Built-in file operations
- **Data Science**: pandas, numpy file operations
- **Machine Learning**: Model save/load operations
- **Visualization**: Plot saving operations
- **Archive Operations**: tar, zip file handling

### Step Type Awareness

Enhanced analysis with step type detection:

- **Registry Integration**: Uses step type registry for classification
- **Framework Detection**: Identifies ML frameworks from imports
- **Pattern Specialization**: Step-specific pattern detection
- **Training Patterns**: Specialized training script analysis

### Context-Rich Reporting

Detailed context preservation for all findings:

- **Line Numbers**: Precise location of all patterns
- **Code Context**: Surrounding code for better understanding
- **Method Attribution**: Identifies the method used for operations
- **Relationship Tracking**: Links related patterns and operations

## Error Handling

Comprehensive error handling throughout analysis:

1. **File Reading**: Graceful handling of file access issues
2. **AST Parsing**: Recovery from syntax errors and malformed code
3. **Pattern Extraction**: Continues analysis when specific patterns fail
4. **Context Extraction**: Provides fallback when context is unavailable

## Performance Considerations

Optimized for large script analysis:

- **Single AST Parse**: Reuses parsed AST across all analysis methods
- **Efficient Traversal**: Optimized visitor patterns for AST traversal
- **Memory Management**: Efficient handling of large script files
- **Lazy Evaluation**: On-demand analysis of specific pattern types

## Testing and Validation

The analyzer supports comprehensive testing:

- **Mock Scripts**: Can analyze synthetic script content
- **Pattern Testing**: Validates pattern detection accuracy
- **AST Testing**: Tests AST parsing and traversal
- **Integration Testing**: Validates integration with validation framework

## Future Enhancements

Potential improvements for the analyzer:

1. **Enhanced Patterns**: Support for more complex code patterns
2. **Type Analysis**: Variable type inference and tracking
3. **Control Flow**: Analysis of conditional and loop structures
4. **Performance Metrics**: Analysis performance tracking
5. **Custom Patterns**: Configurable pattern detection rules

## Conclusion

The ScriptAnalyzer provides comprehensive static analysis of Python scripts using AST parsing, serving as the foundation for alignment validation and script compliance checking. Its multi-dimensional analysis approach ensures thorough extraction of usage patterns while maintaining context and supporting step type-aware validation.

The analyzer serves as a critical component in the validation framework, enabling automated detection of script patterns and supporting comprehensive alignment validation across the entire system.

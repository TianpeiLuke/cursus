---
tags:
  - code
  - validation
  - alignment
  - static_analysis
  - path_analysis
keywords:
  - path extractor
  - static analysis
  - file paths
  - sagemaker paths
  - path construction
  - file operations
  - path validation
topics:
  - validation framework
  - static code analysis
  - path management
language: python
date of note: 2025-08-19
---

# Path Extractor

## Overview

The `PathExtractor` class provides specialized analysis of path usage patterns in Python scripts, with particular focus on SageMaker container path conventions. It identifies hardcoded paths, dynamic path construction patterns, file operations, and potential path usage inconsistencies to support alignment validation.

## Architecture

### Core Capabilities

1. **Hardcoded Path Detection**: Identifies literal path strings in script content
2. **Dynamic Path Construction**: Analyzes `os.path.join()` and `pathlib.Path` usage
3. **File Operation Analysis**: Tracks file read/write operations and their paths
4. **SageMaker Path Validation**: Specialized analysis for SageMaker container paths
5. **Inconsistency Detection**: Identifies mixed path construction patterns and violations

### Analysis Scope

The extractor provides comprehensive path analysis across multiple dimensions:

- **Static Paths**: Hardcoded string literals
- **Dynamic Paths**: Runtime path construction
- **File Operations**: Read/write operations with path context
- **SageMaker Compliance**: Container path convention validation
- **Consistency Checking**: Mixed construction pattern detection

## Implementation Details

### Class Structure

```python
class PathExtractor:
    """
    Specialized extractor for path usage patterns in scripts.
    
    Identifies:
    - Hardcoded path strings
    - Path construction using os.path.join()
    - Path manipulation using pathlib
    - File operations (open, read, write)
    """
```

### Key Methods

#### `extract_hardcoded_paths() -> List[str]`

Identifies literal path strings using comprehensive pattern matching:

**Supported Path Formats:**
- SageMaker paths: `/opt/ml/processing/input/data.csv`
- Absolute Unix paths: `/home/user/data/file.txt`
- Relative paths: `./data/input.csv`, `../config/settings.json`
- Windows paths: `C:\Users\data\file.txt`

**Features:**
- Regex-based pattern matching for multiple path formats
- Duplicate removal while preserving order
- Path likelihood validation to filter false positives
- Context-aware extraction from string literals

#### `extract_path_constructions() -> List[PathConstruction]`

Analyzes dynamic path construction patterns:

**Supported Construction Methods:**
- `os.path.join()`: Traditional path joining
- `pathlib.Path()`: Modern path object construction
- `PurePath()`: Platform-independent path construction

**PathConstruction Object:**
```python
PathConstruction(
    base_path: str,
    construction_parts: List[str],
    line_number: int,
    context: str,
    method: str  # 'os.path.join' or 'pathlib.Path'
)
```

#### `extract_file_operations() -> List[FileOperation]`

Identifies file operations and their associated paths:

**Supported Operations:**
- `open()` function calls
- `with open()` context managers
- File mode detection (read/write/append)

**FileOperation Object:**
```python
FileOperation(
    file_path: str,
    operation_type: str,  # 'read', 'write', 'unknown'
    line_number: int,
    context: str,
    mode: str  # File mode string
)
```

#### `analyze_sagemaker_path_usage() -> Dict[str, List[str]]`

Specialized analysis for SageMaker container paths:

**Path Categories:**
- **Input Paths**: `/opt/ml/processing/input/`, `/opt/ml/input/data/`
- **Output Paths**: `/opt/ml/processing/output/`, `/opt/ml/output/`
- **Model Paths**: `/opt/ml/model/`
- **Other Paths**: Other SageMaker-specific paths

**Return Structure:**
```python
{
    'input_paths': ['/opt/ml/processing/input/train.csv'],
    'output_paths': ['/opt/ml/processing/output/results.json'],
    'model_paths': ['/opt/ml/model/model.pkl'],
    'other_paths': ['/opt/ml/code/script.py']
}
```

#### `extract_logical_names_from_paths() -> Dict[str, str]`

Extracts logical names from SageMaker paths for validation:

- Maps logical names to their corresponding paths
- Supports SageMaker naming conventions
- Enables cross-validation with contract specifications

#### `find_path_inconsistencies() -> List[Dict[str, Any]]`

Identifies potential path usage issues:

**Inconsistency Types:**
- **Mixed Construction**: Same path both hardcoded and dynamically constructed
- **Non-SageMaker Paths**: Non-container paths in SageMaker scripts
- **Path Convention Violations**: Inconsistent path patterns

**Inconsistency Report:**
```python
{
    'type': 'mixed_construction',
    'path': '/opt/ml/input/data/train.csv',
    'issue': 'Path appears both hardcoded and dynamically constructed',
    'recommendation': 'Use consistent path construction method'
}
```

#### `get_path_summary() -> Dict[str, Any]`

Provides comprehensive path usage summary:

```python
{
    'hardcoded_paths': ['/opt/ml/input/data/train.csv', ...],
    'path_constructions': 5,
    'file_operations': 8,
    'sagemaker_paths': {
        'input_paths': 3,
        'output_paths': 2,
        'model_paths': 1,
        'other_paths': 0
    },
    'logical_names': ['train', 'test', 'model'],
    'inconsistencies': 1,
    'total_paths_found': 12
}
```

## Usage Examples

### Basic Path Analysis

```python
from cursus.validation.alignment.static_analysis.path_extractor import PathExtractor

# Initialize with script content
script_content = """
import os
import pandas as pd

# Hardcoded paths
train_path = "/opt/ml/processing/input/train.csv"
output_dir = "/opt/ml/processing/output/"

# Dynamic path construction
model_path = os.path.join("/opt/ml/model", "model.pkl")
result_path = os.path.join(output_dir, "results.json")

# File operations
with open(train_path, 'r') as f:
    data = pd.read_csv(f)

with open(result_path, 'w') as f:
    f.write(json.dumps(results))
"""

script_lines = script_content.split('\n')
extractor = PathExtractor(script_content, script_lines)
```

### Hardcoded Path Detection

```python
# Extract hardcoded paths
hardcoded_paths = extractor.extract_hardcoded_paths()

for path in hardcoded_paths:
    print(f"Hardcoded path: {path}")
```

### Path Construction Analysis

```python
# Analyze dynamic path construction
constructions = extractor.extract_path_constructions()

for construction in constructions:
    print(f"Line {construction.line_number}: {construction.method}")
    print(f"  Base: {construction.base_path}")
    print(f"  Parts: {construction.construction_parts}")
```

### File Operation Analysis

```python
# Analyze file operations
operations = extractor.extract_file_operations()

for operation in operations:
    print(f"Line {operation.line_number}: {operation.operation_type}")
    print(f"  Path: {operation.file_path}")
    print(f"  Mode: {operation.mode}")
```

### SageMaker Path Analysis

```python
# Analyze SageMaker-specific paths
sagemaker_analysis = extractor.analyze_sagemaker_path_usage()

print(f"Input paths: {sagemaker_analysis['input_paths']}")
print(f"Output paths: {sagemaker_analysis['output_paths']}")
print(f"Model paths: {sagemaker_analysis['model_paths']}")
```

### Inconsistency Detection

```python
# Find path inconsistencies
inconsistencies = extractor.find_path_inconsistencies()

for issue in inconsistencies:
    print(f"Issue: {issue['type']}")
    print(f"Path: {issue['path']}")
    print(f"Problem: {issue['issue']}")
    print(f"Recommendation: {issue['recommendation']}")
```

### Comprehensive Analysis

```python
# Get complete path summary
summary = extractor.get_path_summary()

print(f"Total paths found: {summary['total_paths_found']}")
print(f"Hardcoded paths: {len(summary['hardcoded_paths'])}")
print(f"SageMaker input paths: {summary['sagemaker_paths']['input_paths']}")
print(f"Inconsistencies: {summary['inconsistencies']}")
```

## Integration Points

### Alignment Validation Framework

The PathExtractor integrates with the alignment validation system:

```python
class ScriptValidator:
    def validate_paths(self, script_path):
        # Read script content
        with open(script_path, 'r') as f:
            content = f.read()
        lines = content.split('\n')
        
        # Extract and analyze paths
        extractor = PathExtractor(content, lines)
        
        # Validate SageMaker compliance
        sagemaker_analysis = extractor.analyze_sagemaker_path_usage()
        inconsistencies = extractor.find_path_inconsistencies()
        
        # Check against contract requirements
        logical_names = extractor.extract_logical_names_from_paths()
        
        return {
            'sagemaker_compliance': len(inconsistencies) == 0,
            'logical_names': logical_names,
            'path_issues': inconsistencies
        }
```

### Static Analysis Pipeline

Works as part of comprehensive static analysis:

- **AST Integration**: Receives script content from AST parsing
- **Contract Validation**: Validates paths against contract specifications
- **Compliance Checking**: Ensures SageMaker path convention adherence
- **Report Generation**: Provides detailed path analysis reports

### SageMaker Integration

Specialized support for SageMaker container environments:

- **Container Path Validation**: Ensures proper `/opt/ml/` path usage
- **Logical Name Extraction**: Maps paths to logical input/output names
- **Convention Compliance**: Validates against SageMaker best practices
- **Cross-Validation**: Supports validation against processing job contracts

## Advanced Features

### Path Likelihood Detection

Sophisticated filtering to identify actual file paths:

- **Format Recognition**: Identifies various path formats and conventions
- **False Positive Filtering**: Excludes URLs, environment variables, and non-path strings
- **Extension Detection**: Recognizes file extensions as path indicators
- **Context Analysis**: Uses surrounding code context for validation

### Dynamic Path Resolution

Attempts to resolve dynamically constructed paths:

- **Static Analysis**: Resolves paths when all components are literals
- **Variable Tracking**: Identifies path-related variables
- **Expression Parsing**: Handles simple path construction expressions
- **Fallback Handling**: Graceful handling of unresolvable paths

### SageMaker Convention Validation

Comprehensive SageMaker path validation:

- **Container Path Recognition**: Identifies standard SageMaker paths
- **Input/Output Classification**: Categorizes paths by function
- **Logical Name Mapping**: Extracts meaningful names from paths
- **Best Practice Enforcement**: Validates against SageMaker conventions

## Error Handling

The extractor implements robust error handling:

1. **Regex Failures**: Gracefully handles pattern matching errors
2. **Path Resolution**: Continues analysis when path construction fails
3. **Content Parsing**: Handles malformed or incomplete expressions
4. **Context Extraction**: Provides fallback when line context is unavailable

## Performance Considerations

Optimized for large script analysis:

- **Single-Pass Analysis**: Efficient content scanning
- **Pattern Compilation**: Pre-compiled regex patterns
- **Memory Management**: Efficient handling of large script files
- **Lazy Evaluation**: On-demand analysis of specific path types

## Testing and Validation

The extractor supports comprehensive testing:

- **Pattern Testing**: Validates regex pattern accuracy
- **Path Recognition**: Tests path likelihood detection
- **Construction Analysis**: Verifies dynamic path resolution
- **SageMaker Compliance**: Tests container path validation

## Future Enhancements

Potential improvements for the extractor:

1. **Advanced Resolution**: Enhanced dynamic path resolution
2. **Cross-File Analysis**: Path usage across multiple files
3. **Performance Metrics**: Path analysis performance tracking
4. **Custom Patterns**: Configurable path pattern recognition
5. **Integration APIs**: Enhanced integration with validation frameworks

## Conclusion

The PathExtractor provides specialized static analysis of file path usage patterns in Python scripts, with particular strength in SageMaker container path validation. Its comprehensive analysis capabilities support the alignment validation framework by ensuring scripts follow proper path conventions and identifying potential issues before deployment.

The extractor serves as a critical component in maintaining path consistency and SageMaker compliance across the validation framework, enabling automated detection of path-related issues and convention violations.

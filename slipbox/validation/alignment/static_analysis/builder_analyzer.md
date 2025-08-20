---
tags:
  - code
  - validation
  - alignment
  - static_analysis
  - builder_analysis
keywords:
  - builder analyzer
  - argument extraction
  - AST parsing
  - step builder analysis
  - job arguments
  - builder registry
  - script mapping
  - command line arguments
topics:
  - alignment validation
  - static analysis
  - builder analysis
  - argument validation
language: python
date of note: 2025-08-19
---

# Builder Analyzer

## Overview

The Builder Analyzer provides AST-based analysis of step builder classes to extract command-line arguments from `_get_job_arguments()` methods. This enables validation of config-driven arguments that are provided by builders but may not be declared in script contracts, supporting comprehensive argument alignment validation.

## Core Components

### BuilderArgumentExtractor Class

Extracts command-line arguments from step builder files using AST parsing.

#### Initialization

```python
def __init__(self, builder_file_path: str)
```

Initializes the extractor with a builder file path and automatically parses the file into an AST for analysis.

#### Key Methods

```python
def extract_job_arguments(self) -> Set[str]
```

Extracts command-line arguments from the `_get_job_arguments()` method:
- Finds the `_get_job_arguments` method in the AST
- Extracts argument names from string literals
- Returns set of argument names (without `--` prefix)
- Handles both modern and legacy AST node types

```python
def get_method_source(self) -> Optional[str]
```

Returns the source code of the `_get_job_arguments` method for debugging purposes.

### BuilderRegistry Class

Registry for mapping script names to their corresponding step builders.

#### Initialization

```python
def __init__(self, builders_dir: str)
```

Initializes the registry with a builders directory and automatically builds script-to-builder mappings.

#### Key Methods

```python
def get_builder_for_script(self, script_name: str) -> Optional[str]
```

Returns the builder file path for a given script name.

```python
def get_all_mappings(self) -> Dict[str, str]
```

Returns all script-to-builder mappings for debugging and analysis.

## Implementation Details

### AST-Based Argument Extraction

The extractor uses sophisticated AST parsing to find arguments:

```python
def _extract_arguments_from_method(self, method_node: ast.FunctionDef) -> Set[str]:
    arguments = set()
    
    for node in ast.walk(method_node):
        # Handle string constants (Python 3.8+)
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith('--'):
                arg_name = node.value[2:]  # Remove -- prefix
                arguments.add(arg_name)
        
        # Handle legacy ast.Str nodes (Python < 3.8)
        elif isinstance(node, ast.Str) and node.s.startswith('--'):
            arg_name = node.s[2:]  # Remove -- prefix
            arguments.add(arg_name)
        
        # Handle list literals containing arguments
        elif isinstance(node, ast.List):
            for element in node.elts:
                # Extract arguments from list elements
                # ...
    
    return arguments
```

### Script-to-Builder Mapping

The registry uses multiple strategies to map scripts to builders:

#### Strategy 1: Filename Pattern Matching
- Pattern: `builder_<script_name>_step.py`
- Extracts script name from filename structure
- Handles standard naming conventions

#### Strategy 2: Config Import Analysis
- Searches for config imports: `from ..configs.config_<script>_step import`
- Extracts script names from import statements
- Links builders to their configuration dependencies

#### Strategy 3: Entry Point References
- Searches for entry point patterns: `entry_point.*"<script>.py"`
- Extracts script names from entry point declarations
- Identifies script-builder relationships

#### Strategy 4: Name Variation Generation
Handles common naming variations:
- `preprocessing` ↔ `preprocess`
- `evaluation` ↔ `eval`
- `xgboost` ↔ `xgb`

### Error Handling

The analyzer provides robust error handling:
- **File Parsing Errors**: Graceful handling of malformed Python files
- **Missing Methods**: Returns empty sets for builders without `_get_job_arguments`
- **AST Parsing Failures**: Continues processing other builders
- **Import Errors**: Handles missing or invalid imports

## Usage Examples

### Basic Argument Extraction

```python
# Extract arguments from a specific builder
extractor = BuilderArgumentExtractor('builder_preprocessing_step.py')
arguments = extractor.extract_job_arguments()
print(f"Builder provides arguments: {arguments}")

# Example output: {'input-data', 'output-data', 'config-file', 'verbose'}
```

### Builder Registry Usage

```python
# Initialize registry
registry = BuilderRegistry('src/cursus/steps/builders')

# Find builder for script
builder_file = registry.get_builder_for_script('preprocessing')
print(f"Builder file: {builder_file}")

# Get all mappings
mappings = registry.get_all_mappings()
for script, builder in mappings.items():
    print(f"{script} -> {builder}")
```

### Convenience Function

```python
# Extract arguments using convenience function
arguments = extract_builder_arguments('preprocessing', 'src/cursus/steps/builders')
print(f"Arguments provided by builder: {arguments}")
```

### Integration with Validation

```python
# Use in validation context
def validate_script_arguments(script_name, script_arguments, builders_dir):
    # Get builder-provided arguments
    builder_args = extract_builder_arguments(script_name, builders_dir)
    
    # Check for undeclared arguments that are provided by builder
    undeclared_args = script_arguments - builder_args
    
    if undeclared_args:
        print(f"Script uses undeclared arguments: {undeclared_args}")
    
    # Check for unused builder arguments
    unused_builder_args = builder_args - script_arguments
    
    if unused_builder_args:
        print(f"Builder provides unused arguments: {unused_builder_args}")
```

## Integration Points

### Script Contract Validator

Integrates with script-contract validation to:
- Identify builder-provided arguments
- Reduce false positives for undeclared argument access
- Validate argument consistency between scripts and builders
- Support comprehensive argument alignment validation

### Alignment Validation System

Provides builder analysis for:
- Level 4 (Builder-Configuration) alignment validation
- Builder pattern recognition and validation
- Configuration field validation enhancement
- Step builder consistency checking

### Pattern Recognizer

Works with pattern recognition to:
- Filter false positives for builder-provided arguments
- Recognize legitimate builder argument patterns
- Support architectural pattern validation
- Enable builder-aware validation rules

### Static Analysis Framework

Integrates with static analysis for:
- Comprehensive code analysis workflows
- Multi-component validation coordination
- AST-based analysis consistency
- Validation result aggregation

## Benefits

### Comprehensive Argument Analysis
- Extracts all builder-provided arguments automatically
- Handles complex argument patterns and structures
- Supports various argument declaration styles
- Provides complete argument coverage

### False Positive Reduction
- Identifies legitimate builder-provided arguments
- Reduces validation noise from expected arguments
- Improves validation accuracy and usefulness
- Focuses attention on actual issues

### Flexible Builder Discovery
- Supports multiple builder discovery strategies
- Handles various naming conventions and patterns
- Provides robust script-to-builder mapping
- Adapts to different project structures

### AST-Based Accuracy
- Uses precise AST parsing for argument extraction
- Handles complex Python code structures
- Supports both modern and legacy Python versions
- Provides reliable and accurate analysis

## Advanced Features

### Name Variation Handling

The registry handles common naming variations:

```python
def _generate_name_variations(self, name: str) -> List[str]:
    variations = []
    
    # Handle preprocessing variations
    if 'preprocessing' in name:
        variations.append(name.replace('preprocessing', 'preprocess'))
    if 'preprocess' in name and 'preprocessing' not in name:
        variations.append(name.replace('preprocess', 'preprocessing'))
    
    # Handle evaluation variations
    if 'evaluation' in name:
        variations.append(name.replace('evaluation', 'eval'))
    if 'eval' in name and 'evaluation' not in name:
        variations.append(name.replace('eval', 'evaluation'))
    
    return variations
```

### Multi-Strategy Script Discovery

Uses multiple strategies for robust script-builder mapping:

```python
def _extract_script_names_from_builder(self, builder_file: Path) -> List[str]:
    script_names = []
    
    # Strategy 1: Filename pattern
    if filename.startswith('builder_') and filename.endswith('_step'):
        middle_part = filename[8:-5]  # Extract script name
        script_names.append(middle_part)
    
    # Strategy 2: Config imports
    config_import_pattern = r'from\s+\.\.configs\.config_(\w+)_step\s+import'
    matches = re.findall(config_import_pattern, content)
    script_names.extend(matches)
    
    # Strategy 3: Entry point references
    entry_point_pattern = r'entry_point.*["\'](\w+)\.py["\']'
    matches = re.findall(entry_point_pattern, content)
    script_names.extend(matches)
    
    return list(set(script_names))
```

### Debugging Support

Provides debugging capabilities:

```python
# Get method source for debugging
extractor = BuilderArgumentExtractor('builder_file.py')
method_source = extractor.get_method_source()
print("Method source:")
print(method_source)

# Get all registry mappings
registry = BuilderRegistry('builders_dir')
mappings = registry.get_all_mappings()
print("All script-to-builder mappings:")
for script, builder in mappings.items():
    print(f"  {script} -> {builder}")
```

## Performance Considerations

### Efficient AST Parsing
- Parses each builder file only once
- Caches AST structures for reuse
- Uses efficient AST traversal algorithms
- Minimizes memory usage during analysis

### Registry Optimization
- Builds mappings once during initialization
- Caches script-to-builder relationships
- Uses efficient lookup structures
- Supports batch processing operations

### Scalability
- Handles large numbers of builder files efficiently
- Supports parallel processing for batch analysis
- Optimized for repeated argument extraction
- Memory-conscious handling of large codebases

## Error Recovery

The analyzer provides comprehensive error recovery:
- **Parse Errors**: Continues with other builders when one fails
- **Missing Methods**: Returns empty sets gracefully
- **Invalid Syntax**: Handles malformed Python files
- **Import Failures**: Processes available builders despite failures

## Future Enhancements

### Planned Improvements
- Support for dynamic argument generation
- Enhanced pattern recognition for complex builders
- Integration with IDE tooling for real-time analysis
- Support for custom argument extraction patterns
- Advanced caching and memoization
- Integration with external analysis tools
- Support for builder inheritance analysis
- Enhanced debugging and diagnostic capabilities

## Conclusion

The Builder Analyzer provides essential functionality for comprehensive argument validation in the alignment validation system. By extracting builder-provided arguments through AST analysis, it enables accurate validation while reducing false positives and improving the overall validation experience.

---
tags:
  - code
  - validation
  - alignment
  - builder
  - analysis
keywords:
  - builder code analysis
  - AST parsing
  - configuration usage
  - validation patterns
  - code analysis
  - builder patterns
  - architectural analysis
  - static analysis
topics:
  - validation framework
  - code analysis
  - builder validation
  - static analysis
language: python
date of note: 2025-08-18
---

# Builder Code Analyzer

## Overview

The Builder Code Analyzer provides comprehensive static analysis of step builder code using Abstract Syntax Tree (AST) parsing. It extracts configuration usage patterns, validation calls, and architectural information to support Level 4 alignment validation.

## Core Components

### BuilderCodeAnalyzer Class

The main analyzer class that orchestrates AST-based code analysis:

```python
class BuilderCodeAnalyzer:
    """
    Analyzes builder code to extract configuration usage patterns and architectural information.
    
    Uses AST parsing to identify:
    - Configuration field accesses
    - Validation method calls
    - Default value assignments
    - Class and method definitions
    """
```

### File Analysis Interface

```python
def analyze_builder_file(self, builder_path: Path) -> Dict[str, Any]:
    """
    Analyze builder file to extract code patterns.
    
    Args:
        builder_path: Path to the builder file
        
    Returns:
        Dictionary containing builder analysis results
    """
```

**Analysis Process:**
1. **File Reading**: Load builder source code
2. **AST Parsing**: Parse code into Abstract Syntax Tree
3. **Pattern Extraction**: Extract architectural patterns
4. **Result Compilation**: Compile analysis results

**Error Handling:**
```python
except Exception as e:
    return {
        'error': str(e),
        'config_accesses': [],
        'validation_calls': [],
        'default_assignments': [],
        'class_definitions': [],
        'method_definitions': []
    }
```

### AST Analysis Engine

```python
def analyze_builder_code(self, builder_ast: ast.AST, builder_content: str) -> Dict[str, Any]:
    """Analyze builder AST to extract configuration usage patterns."""
```

**Analysis Categories:**
- **Configuration Accesses**: Field access patterns
- **Validation Calls**: Validation method invocations
- **Default Assignments**: Default value assignments
- **Class Definitions**: Class structure analysis
- **Method Definitions**: Method signature analysis
- **Import Statements**: Import pattern analysis
- **Configuration Class Usage**: Config class utilization

## AST Visitor Implementation

### BuilderVisitor Class

Specialized AST visitor for extracting builder-specific patterns:

```python
class BuilderVisitor(ast.NodeVisitor):
    """AST visitor for analyzing builder code patterns."""
```

### Method Call Analysis

```python
def visit_Call(self, node):
    """Visit function/method call nodes."""
```

**Call Pattern Detection:**
- **Configuration Method Calls**: `config.method()` patterns
- **Self Configuration Calls**: `self.config.method()` patterns
- **Validation Method Calls**: Validation framework usage
- **Method Call Tracking**: Distinguish methods from field accesses

**Validation Method Detection:**
```python
if node.func.attr in ['validate', 'require', 'check', 'assert_required']:
    self.analysis['validation_calls'].append({
        'method': node.func.attr,
        'line_number': node.lineno,
        'args': len(node.args),
        'context': self._get_context(node)
    })
```

### Attribute Access Analysis

```python
def visit_Attribute(self, node):
    """Visit attribute access nodes (e.g., config.field_name or self.config.field_name)."""
```

**Access Pattern Detection:**
- **Direct Config Access**: `config.field_name`
- **Self Config Access**: `self.config.field_name`
- **Method Call Exclusion**: Filter out method calls from field accesses
- **Context Tracking**: Record access location and context

**Field Access Recording:**
```python
if (isinstance(node.value, ast.Name) and 
    node.value.id == 'config'):
    # Only record as field access if it's not a method call
    if (node.attr, node.lineno) not in self.method_calls:
        self.analysis['config_accesses'].append({
            'field_name': node.attr,
            'line_number': node.lineno,
            'context': self._get_context(node)
        })
```

### Assignment Analysis

```python
def visit_Assign(self, node):
    """Visit assignment nodes."""
```

**Assignment Pattern Detection:**
- **Default Value Assignments**: Field default value setting
- **Target Analysis**: Assignment target identification
- **Context Recording**: Assignment location and context

### Definition Analysis

```python
def visit_ClassDef(self, node):
    """Visit class definition nodes."""

def visit_FunctionDef(self, node):
    """Visit function/method definition nodes."""

def visit_AsyncFunctionDef(self, node):
    """Visit async function/method definition nodes."""
```

**Definition Information Extraction:**
- **Class Definitions**: Class names, base classes, decorators
- **Method Definitions**: Method names, arguments, decorators
- **Async Method Support**: Async function detection
- **Inheritance Analysis**: Base class identification

### Import Analysis

```python
def visit_Import(self, node):
    """Visit import statements."""

def visit_ImportFrom(self, node):
    """Visit from...import statements."""
```

**Import Pattern Detection:**
- **Standard Imports**: `import module` statements
- **From Imports**: `from module import name` statements
- **Alias Tracking**: Import alias detection
- **Module Analysis**: Imported module identification

## Pattern Analysis Engine

### BuilderPatternAnalyzer Class

Higher-level pattern analysis beyond basic AST parsing:

```python
class BuilderPatternAnalyzer:
    """
    Analyzes builder patterns and architectural compliance.
    
    Provides higher-level analysis of builder code patterns beyond basic AST parsing.
    """
```

### Configuration Usage Analysis

```python
def analyze_configuration_usage(self, builder_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze how configuration is used in the builder."""
```

**Usage Pattern Analysis:**
- **Field Grouping**: Group accesses by field name
- **Access Counting**: Count field access frequency
- **Line Range Analysis**: First and last access locations
- **Context Analysis**: Access context identification

**Usage Metrics:**
```python
usage_patterns[field_name] = {
    'access_count': len(accesses),
    'first_access_line': min(access['line_number'] for access in accesses),
    'last_access_line': max(access['line_number'] for access in accesses),
    'contexts': [access['context'] for access in accesses]
}
```

**Analysis Results:**
- **Accessed Fields**: Set of all accessed configuration fields
- **Field Usage**: Detailed usage information per field
- **Usage Patterns**: Access pattern analysis
- **Total Accesses**: Overall configuration access count

### Validation Pattern Analysis

```python
def analyze_validation_patterns(self, builder_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze validation patterns in the builder."""
```

**Validation Analysis:**
- **Validation Detection**: Presence of validation logic
- **Method Categorization**: Group validation calls by method type
- **Call Counting**: Validation method usage frequency
- **Location Tracking**: Validation call line numbers

**Analysis Results:**
```python
return {
    'has_validation': len(validation_calls) > 0,
    'validation_methods': validation_methods,
    'validation_call_count': len(validation_calls),
    'validation_lines': [call['line_number'] for call in validation_calls]
}
```

### Import Pattern Analysis

```python
def analyze_import_patterns(self, builder_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze import patterns to detect configuration class usage."""
```

**Import Analysis:**
- **Configuration Import Detection**: Identify config-related imports
- **Module Analysis**: Analyze imported modules
- **Import Counting**: Total import statement count
- **Config Usage Detection**: Configuration class import presence

**Configuration Import Detection:**
```python
config_imports = []
for stmt in import_statements:
    if 'config' in stmt.get('module', '').lower() or 'config' in stmt.get('name', '').lower():
        config_imports.append(stmt)
```

## Analysis Results Structure

### Core Analysis Output

```python
analysis = {
    'config_accesses': [
        {
            'field_name': 'input_path',
            'line_number': 42,
            'context': 'line_42'
        }
    ],
    'validation_calls': [
        {
            'method': 'validate',
            'line_number': 35,
            'args': 2,
            'context': 'line_35'
        }
    ],
    'default_assignments': [
        {
            'field_name': 'batch_size',
            'line_number': 28,
            'target_type': 'Attribute',
            'context': 'line_28'
        }
    ],
    'class_definitions': [
        {
            'class_name': 'ProcessingStepBuilder',
            'line_number': 15,
            'base_classes': ['BaseStepBuilder'],
            'decorators': []
        }
    ],
    'method_definitions': [
        {
            'method_name': 'build_step',
            'line_number': 45,
            'args': ['self', 'config'],
            'decorators': [],
            'is_async': False
        }
    ],
    'import_statements': [
        {
            'type': 'from_import',
            'module': 'config.processing_config',
            'name': 'ProcessingConfig',
            'alias': None,
            'line_number': 5
        }
    ]
}
```

### Pattern Analysis Output

```python
configuration_usage = {
    'accessed_fields': {'input_path', 'output_path', 'batch_size'},
    'field_usage': {
        'input_path': [
            {'field_name': 'input_path', 'line_number': 42, 'context': 'line_42'}
        ]
    },
    'usage_patterns': {
        'input_path': {
            'access_count': 1,
            'first_access_line': 42,
            'last_access_line': 42,
            'contexts': ['line_42']
        }
    },
    'total_config_accesses': 3
}

validation_patterns = {
    'has_validation': True,
    'validation_methods': {
        'validate': [
            {'method': 'validate', 'line_number': 35, 'args': 2, 'context': 'line_35'}
        ]
    },
    'validation_call_count': 1,
    'validation_lines': [35]
}

import_patterns = {
    'total_imports': 5,
    'config_imports': [
        {
            'type': 'from_import',
            'module': 'config.processing_config',
            'name': 'ProcessingConfig',
            'alias': None,
            'line_number': 5
        }
    ],
    'has_config_import': True,
    'import_modules': ['pathlib', 'typing', 'config.processing_config']
}
```

## Integration with Alignment Validation

### Builder Configuration Alignment

The analyzer integrates with BuilderConfigurationAlignmentTester:

```python
# Analyze builder code using extracted component
builder_analysis = self.builder_analyzer.analyze_builder_file(builder_path)

# Validate configuration field handling
config_issues = self._validate_configuration_fields(builder_analysis, config_analysis, builder_name)
```

### Field Access Validation

```python
# Get fields accessed in builder
accessed_fields = set()
for access in builder_analysis.get('config_accesses', []):
    accessed_fields.add(access['field_name'])

# Check for accessed fields not in configuration
undeclared_fields = accessed_fields - config_fields
```

### Validation Logic Detection

```python
# Check if builder has validation logic
has_validation = len(builder_analysis.get('validation_calls', [])) > 0

if required_fields and not has_validation:
    # Report missing validation for required fields
```

## Utility Functions

### Context Extraction

```python
def _get_context(self, node) -> str:
    """Get context information for a node (e.g., which method it's in)."""
```

**Context Information:**
- **Line Number**: Source code line location
- **Method Context**: Current method being analyzed
- **Class Context**: Current class being analyzed

### Name Extraction

```python
def _get_name(self, node) -> str:
    """Extract name from various AST node types."""
```

**Name Extraction Support:**
- **Simple Names**: `ast.Name` nodes
- **Attribute Names**: `ast.Attribute` nodes with dot notation
- **Constant Values**: `ast.Constant` nodes
- **Fallback Handling**: Node type names for unsupported types

## Error Handling

### Robust Analysis

The analyzer includes comprehensive error handling:

- **File Reading Errors**: Handle missing or unreadable files
- **Syntax Errors**: Handle malformed Python code
- **AST Parsing Errors**: Handle parsing failures
- **Analysis Errors**: Handle unexpected code patterns

### Graceful Degradation

```python
try:
    with open(builder_path, 'r') as f:
        builder_content = f.read()
    
    builder_ast = ast.parse(builder_content)
    return self.analyze_builder_code(builder_ast, builder_content)
except Exception as e:
    return {
        'error': str(e),
        'config_accesses': [],
        'validation_calls': [],
        # ... empty results with error indication
    }
```

## Best Practices

### Builder Code Analysis

**Clear Configuration Usage:**
```python
# Good: Direct configuration field access
def build_step(self, config):
    input_path = config.input_path
    batch_size = config.batch_size
```

**Validation Implementation:**
```python
# Good: Explicit validation calls
def validate_config(self, config):
    self.validate(config.input_path, "Input path is required")
    self.require(config.output_path, "Output path is required")
```

### Import Organization

**Configuration Import:**
```python
# Good: Clear configuration import
from config.processing_config import ProcessingConfig
```

**Module Organization:**
```python
# Good: Organized imports
from pathlib import Path
from typing import Dict, Any
from config.processing_config import ProcessingConfig
```

The Builder Code Analyzer provides essential static analysis capabilities for understanding builder implementation patterns and supporting comprehensive Level 4 alignment validation through detailed AST-based code analysis.

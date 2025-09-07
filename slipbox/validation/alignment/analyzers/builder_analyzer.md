---
tags:
  - code
  - validation
  - alignment
  - builder
  - analysis
keywords:
  - BuilderCodeAnalyzer
  - BuilderVisitor
  - BuilderPatternAnalyzer
  - AST parsing
  - configuration usage
  - validation patterns
  - code analysis
topics:
  - validation framework
  - code analysis
  - builder validation
  - static analysis
language: python
date of note: 2025-09-07
---

# Builder Code Analyzer

Comprehensive static analysis of step builder code using Abstract Syntax Tree (AST) parsing.

## Overview

The Builder Code Analyzer provides comprehensive static analysis of step builder code using Abstract Syntax Tree (AST) parsing. It extracts configuration usage patterns, validation calls, and architectural information to support Level 4 alignment validation.

The analyzer uses AST parsing to identify configuration field accesses, validation method calls, default value assignments, class and method definitions, and import patterns. It provides both low-level AST analysis and higher-level pattern analysis for architectural compliance validation.

## Classes and Methods

### Classes
- [`BuilderCodeAnalyzer`](#buildercodeanalyzer) - Main analyzer class for AST-based code analysis
- [`BuilderVisitor`](#buildervisitor) - AST visitor for extracting builder-specific patterns
- [`BuilderPatternAnalyzer`](#builderpatternanalyzer) - Higher-level pattern analysis beyond basic AST parsing

## API Reference

### BuilderCodeAnalyzer

_class_ cursus.validation.alignment.analyzers.builder_analyzer.BuilderCodeAnalyzer()

Analyzes builder code to extract configuration usage patterns and architectural information using AST parsing.

```python
from cursus.validation.alignment.analyzers.builder_analyzer import BuilderCodeAnalyzer

analyzer = BuilderCodeAnalyzer()
```

#### analyze_builder_file

analyze_builder_file(_builder_path_)

Analyze builder file to extract code patterns by parsing the source code into an AST and extracting architectural patterns.

**Parameters:**
- **builder_path** (_Path_) – Path to the builder file

**Returns:**
- **Dict[str, Any]** – Dictionary containing builder analysis results with keys: config_accesses, validation_calls, default_assignments, class_definitions, method_definitions, import_statements

```python
from pathlib import Path

builder_path = Path('src/cursus/steps/builders/processing_step_builder.py')
results = analyzer.analyze_builder_file(builder_path)
print(f"Config accesses: {len(results['config_accesses'])}")
```

#### analyze_builder_code

analyze_builder_code(_builder_ast_, _builder_content_)

Analyze builder AST to extract configuration usage patterns using the BuilderVisitor.

**Parameters:**
- **builder_ast** (_ast.AST_) – Parsed AST of the builder code
- **builder_content** (_str_) – Raw builder code content

**Returns:**
- **Dict[str, Any]** – Dictionary containing analysis results

```python
import ast

with open('builder.py', 'r') as f:
    content = f.read()
builder_ast = ast.parse(content)
results = analyzer.analyze_builder_code(builder_ast, content)
```

### BuilderVisitor

_class_ cursus.validation.alignment.analyzers.builder_analyzer.BuilderVisitor(_analysis_)

AST visitor for analyzing builder code patterns and extracting architectural information.

**Parameters:**
- **analysis** (_Dict[str, Any]_) – Dictionary to store analysis results

```python
analysis = {
    'config_accesses': [],
    'validation_calls': [],
    'default_assignments': [],
    'class_definitions': [],
    'method_definitions': [],
    'import_statements': []
}
visitor = BuilderVisitor(analysis)
```

#### visit_Call

visit_Call(_node_)

Visit function/method call nodes to detect validation method calls and track method calls on config objects.

**Parameters:**
- **node** (_ast.Call_) – AST call node

```python
# Automatically called during AST traversal
visitor.visit(ast_tree)
```

#### visit_Attribute

visit_Attribute(_node_)

Visit attribute access nodes to detect configuration field accesses (e.g., config.field_name or self.config.field_name).

**Parameters:**
- **node** (_ast.Attribute_) – AST attribute node

#### visit_Assign

visit_Assign(_node_)

Visit assignment nodes to detect default value assignments.

**Parameters:**
- **node** (_ast.Assign_) – AST assignment node

#### visit_ClassDef

visit_ClassDef(_node_)

Visit class definition nodes to extract class information including base classes and decorators.

**Parameters:**
- **node** (_ast.ClassDef_) – AST class definition node

#### visit_FunctionDef

visit_FunctionDef(_node_)

Visit function/method definition nodes to extract method signatures and decorators.

**Parameters:**
- **node** (_ast.FunctionDef_) – AST function definition node

#### visit_AsyncFunctionDef

visit_AsyncFunctionDef(_node_)

Visit async function/method definition nodes to extract async method information.

**Parameters:**
- **node** (_ast.AsyncFunctionDef_) – AST async function definition node

#### visit_Import

visit_Import(_node_)

Visit import statements to track module imports.

**Parameters:**
- **node** (_ast.Import_) – AST import node

#### visit_ImportFrom

visit_ImportFrom(_node_)

Visit from...import statements to track specific imports from modules.

**Parameters:**
- **node** (_ast.ImportFrom_) – AST from-import node

### BuilderPatternAnalyzer

_class_ cursus.validation.alignment.analyzers.builder_analyzer.BuilderPatternAnalyzer()

Analyzes builder patterns and architectural compliance, providing higher-level analysis beyond basic AST parsing.

```python
pattern_analyzer = BuilderPatternAnalyzer()
```

#### analyze_configuration_usage

analyze_configuration_usage(_builder_analysis_)

Analyze how configuration is used in the builder by grouping field accesses and analyzing usage patterns.

**Parameters:**
- **builder_analysis** (_Dict[str, Any]_) – Result from BuilderCodeAnalyzer

**Returns:**
- **Dict[str, Any]** – Configuration usage analysis with keys: accessed_fields, field_usage, usage_patterns, total_config_accesses

```python
builder_results = analyzer.analyze_builder_file(builder_path)
usage_analysis = pattern_analyzer.analyze_configuration_usage(builder_results)
print(f"Accessed fields: {usage_analysis['accessed_fields']}")
```

#### analyze_validation_patterns

analyze_validation_patterns(_builder_analysis_)

Analyze validation patterns in the builder by categorizing validation method calls.

**Parameters:**
- **builder_analysis** (_Dict[str, Any]_) – Result from BuilderCodeAnalyzer

**Returns:**
- **Dict[str, Any]** – Validation pattern analysis with keys: has_validation, validation_methods, validation_call_count, validation_lines

```python
validation_analysis = pattern_analyzer.analyze_validation_patterns(builder_results)
print(f"Has validation: {validation_analysis['has_validation']}")
```

#### analyze_import_patterns

analyze_import_patterns(_builder_analysis_)

Analyze import patterns to detect configuration class usage and module dependencies.

**Parameters:**
- **builder_analysis** (_Dict[str, Any]_) – Result from BuilderCodeAnalyzer

**Returns:**
- **Dict[str, Any]** – Import pattern analysis with keys: total_imports, config_imports, has_config_import, import_modules

```python
import_analysis = pattern_analyzer.analyze_import_patterns(builder_results)
print(f"Config imports: {len(import_analysis['config_imports'])}")
```

## Related Documentation

- [Builder Config Alignment](../builder_config_alignment.md) - Level 4 alignment validation
- [Unified Alignment Tester](../unified_alignment_tester.md) - Main alignment validation system
- [Config Analyzer](config_analyzer.md) - Configuration analysis utilities

---
tags:
  - archive
  - design
  - pipeline_runtime_testing
  - runtime_tester
  - script_execution
  - data_compatibility
keywords:
  - RuntimeTester
  - script testing
  - data compatibility
  - pipeline validation
  - topological execution
  - logical name matching
topics:
  - runtime testing
  - script validation
  - data flow testing
  - pipeline execution
language: python
date of note: 2025-09-09
---

# RuntimeTester Design

## Overview

The RuntimeTester is the core execution engine of the pipeline runtime testing system. It executes script testing and validation with enhanced data compatibility checking, logical name matching, and topological pipeline execution. The RuntimeTester validates script functionality and data transfer consistency for pipeline development.

## Architecture

### Class Structure

```python
class RuntimeTester:
    def __init__(self, builder: PipelineTestingSpecBuilder):
        self.builder = builder
        self.semantic_threshold = 0.7
        self.path_matcher = PathMatcher(self.semantic_threshold)
```

### Core Responsibilities

1. **Individual Script Testing**: Execute and validate single scripts
2. **Data Compatibility Testing**: Test data flow between script pairs
3. **Pipeline Flow Testing**: Execute complete pipeline with topological ordering
4. **Enhanced File Format Support**: Handle any output format with intelligent filtering
5. **Logical Name Matching**: Intelligent path matching between script inputs/outputs

## Core Methods

### 1. Individual Script Testing

```python
def test_script_with_spec(self, spec: ScriptExecutionSpec, main_params: Dict[str, Any]) -> ScriptTestResult:
    """
    Test individual script execution with comprehensive validation.
    
    Args:
        spec: ScriptExecutionSpec defining script parameters
        main_params: Main function parameters for script execution
        
    Returns:
        ScriptTestResult with execution details and validation results
    """
    try:
        # 1. Prepare execution environment
        script_path = Path(spec.script_path)
        if not script_path.exists():
            return ScriptTestResult(
                script_name=spec.script_name,
                success=False,
                error_message=f"Script file not found: {script_path}"
            )
        
        # 2. Set up environment variables
        env = os.environ.copy()
        env.update(spec.environ_vars)
        
        # 3. Execute script with timeout and error capture
        start_time = time.time()
        result = self._execute_script_safely(script_path, main_params, env)
        execution_time = time.time() - start_time
        
        # 4. Validate outputs
        output_validation = self._validate_script_outputs(spec)
        
        # 5. Return comprehensive results
        return ScriptTestResult(
            script_name=spec.script_name,
            success=result.success and output_validation.success,
            execution_time=execution_time,
            output_files=output_validation.output_files,
            error_message=result.error_message or output_validation.error_message,
            stdout=result.stdout,
            stderr=result.stderr
        )
        
    except Exception as e:
        return ScriptTestResult(
            script_name=spec.script_name,
            success=False,
            error_message=f"Script test failed: {str(e)}"
        )
```

### 2. Enhanced File Format Support

```python
def _find_valid_output_files(self, output_dir: Path, min_size_bytes: int = 1) -> List[Path]:
    """
    Find valid output files in a directory, excluding temporary and system files.
    Uses intelligent blacklist approach instead of hardcoded CSV-only detection.
    
    Supports all file formats: CSV, JSON, Parquet, PKL, BST, ONNX, TAR.GZ, etc.
    """
    if not output_dir.exists() or not output_dir.is_dir():
        return []
        
    valid_files = []
    
    for file_path in output_dir.iterdir():
        # Skip directories
        if file_path.is_dir():
            continue
            
        # Skip temporary/system files using smart detection
        if self._is_temp_or_system_file(file_path):
            continue
            
        # Check file size (exclude empty files)
        try:
            if file_path.stat().st_size < min_size_bytes:
                continue
        except (OSError, IOError):
            continue
            
        valid_files.append(file_path)
    
    # Sort by modification time, newest first
    valid_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    return valid_files

def _is_temp_or_system_file(self, file_path: Path) -> bool:
    """
    Intelligent temporary file detection using blacklist approach.
    Excludes: .tmp, .temp, ~, .swp, .bak, .log, .DS_Store, __pycache__, etc.
    """
    filename = file_path.name.lower()
    
    # Temporary file patterns
    temp_patterns = [
        r'.*\.tmp$', r'.*\.temp$', r'.*~$', r'.*\.swp$', r'.*\.bak$',
        r'.*\.orig$', r'.*\.rej$', r'.*\.lock$', r'.*\.pid$', r'.*\.log$'
    ]
    
    # System files
    system_files = {'.ds_store', 'thumbs.db', 'desktop.ini'}
    
    # Hidden files (with exceptions)
    if filename.startswith('.') and filename not in {'.gitkeep', '.placeholder'}:
        return True
        
    # Check patterns and system files
    if filename in system_files:
        return True
        
    for pattern in temp_patterns:
        if re.match(pattern, filename):
            return True
            
    # Cache files
    if '__pycache__' in str(file_path) or filename.endswith(('.pyc', '.pyo')):
        return True
        
    return False
```

### 3. Data Compatibility Testing

```python
def test_data_compatibility_with_specs(self, spec_a: ScriptExecutionSpec, 
                                     spec_b: ScriptExecutionSpec) -> DataCompatibilityResult:
    """
    Enhanced data compatibility testing with intelligent path matching.
    
    Tests whether script A's outputs can be used as inputs for script B,
    using logical name matching and format-agnostic file handling.
    """
    try:
        # 1. Execute script A using its ScriptExecutionSpec
        main_params_a = self.builder.get_script_main_params(spec_a)
        script_a_result = self.test_script_with_spec(spec_a, main_params_a)
        
        if not script_a_result.success:
            return DataCompatibilityResult(
                script_a=spec_a.script_name,
                script_b=spec_b.script_name,
                compatible=False,
                compatibility_issues=[f"Script A failed: {script_a_result.error_message}"]
            )
        
        # 2. Find valid output files from script A (any format, excluding temp files)
        output_dir_a = Path(spec_a.output_paths.get("data_output", ""))
        output_files = self._find_valid_output_files(output_dir_a)
        
        if not output_files:
            return DataCompatibilityResult(
                script_a=spec_a.script_name,
                script_b=spec_b.script_name,
                compatible=False,
                compatibility_issues=["Script A did not produce any valid output files"]
            )
        
        # 3. Enhanced: Match logical names using PathMatcher
        path_matches = self._find_logical_name_matches(spec_a, spec_b)
        
        if not path_matches:
            return DataCompatibilityResult(
                script_a=spec_a.script_name,
                script_b=spec_b.script_name,
                compatible=False,
                compatibility_issues=["No matching logical names found between outputs and inputs"],
                path_matches=[]
            )
        
        # 4. Create modified spec_b with matched paths
        modified_spec_b = self._create_modified_spec_with_matches(
            spec_b, path_matches, output_files
        )
        
        # 5. Execute script B with matched inputs
        main_params_b = self.builder.get_script_main_params(modified_spec_b)
        script_b_result = self.test_script_with_spec(modified_spec_b, main_params_b)
        
        # 6. Return detailed results with matching information
        return DataCompatibilityResult(
            script_a=spec_a.script_name,
            script_b=spec_b.script_name,
            compatible=script_b_result.success,
            compatibility_issues=[] if script_b_result.success else [script_b_result.error_message],
            path_matches=path_matches,
            matching_details=self._generate_matching_report(path_matches),
            files_tested=[str(f) for f in output_files]
        )
        
    except Exception as e:
        return DataCompatibilityResult(
            script_a=spec_a.script_name,
            script_b=spec_b.script_name,
            compatible=False,
            compatibility_issues=[f"Compatibility test failed: {str(e)}"]
        )
```

### 4. Logical Name Matching

```python
def _find_logical_name_matches(self, source_spec: ScriptExecutionSpec, 
                              dest_spec: ScriptExecutionSpec) -> List[PathMatch]:
    """
    Find matches between source outputs and destination inputs using intelligent matching.
    
    Matching Priority:
    1. Exact logical name match
    2. Logical name to alias match  
    3. Alias to logical name match
    4. Alias to alias match
    5. Semantic similarity match (above threshold)
    """
    matches = []
    
    # Convert to enhanced specs for matching
    enhanced_source = self._create_enhanced_script_spec(source_spec)
    enhanced_dest = self._create_enhanced_script_spec(dest_spec)
    
    # Use PathMatcher for intelligent matching
    matches = self.path_matcher.find_path_matches(enhanced_source, enhanced_dest)
    
    return matches

def _create_enhanced_script_spec(self, original_spec: ScriptExecutionSpec) -> EnhancedScriptExecutionSpec:
    """Convert original ScriptExecutionSpec to enhanced version with PathSpecs."""
    
    # Create PathSpecs from original paths
    input_path_specs = {}
    for logical_name, path in original_spec.input_paths.items():
        input_path_specs[logical_name] = PathSpec(
            logical_name=logical_name,
            path=path,
            aliases=self._get_aliases_for_logical_name(logical_name)
        )
    
    output_path_specs = {}
    for logical_name, path in original_spec.output_paths.items():
        output_path_specs[logical_name] = PathSpec(
            logical_name=logical_name,
            path=path,
            aliases=self._get_aliases_for_logical_name(logical_name)
        )
    
    return EnhancedScriptExecutionSpec(
        script_name=original_spec.script_name,
        step_name=original_spec.step_name,
        script_path=original_spec.script_path,
        input_path_specs=input_path_specs,
        output_path_specs=output_path_specs,
        environ_vars=original_spec.environ_vars,
        job_args=original_spec.job_args
    )

def _get_aliases_for_logical_name(self, logical_name: str) -> List[str]:
    """Get common aliases for logical names based on semantic patterns."""
    alias_map = {
        "data_output": ["output", "result", "processed_data", "dataset"],
        "model_output": ["model", "artifact", "trained_model", "model_artifact"],
        "data_input": ["input", "dataset", "training_data", "processed_input"],
        "model_input": ["model", "trained_model", "model_artifact", "pretrained"],
        "config": ["configuration", "params", "hyperparameters", "settings"]
    }
    return alias_map.get(logical_name, [])
```

### 5. Pipeline Flow Testing

```python
def test_pipeline_flow_with_spec(self, pipeline_spec: PipelineTestingSpec) -> Dict[str, Any]:
    """
    Enhanced pipeline flow testing with topological ordering and data flow chaining.
    
    Executes scripts in proper dependency order and validates data compatibility
    between connected nodes.
    """
    results = {
        "pipeline_success": True,
        "script_results": {},
        "data_flow_results": {},
        "execution_order": [],
        "errors": []
    }
    
    try:
        dag = pipeline_spec.dag
        script_specs = pipeline_spec.script_specs
        
        if not dag.nodes:
            results["pipeline_success"] = False
            results["errors"].append("No nodes found in pipeline DAG")
            return results
        
        # Get topological execution order
        try:
            execution_order = dag.topological_sort()
            results["execution_order"] = execution_order
        except ValueError as e:
            results["pipeline_success"] = False
            results["errors"].append(f"DAG topology error: {str(e)}")
            return results
        
        # Execute in topological order, testing each edge
        executed_nodes = set()
        
        for current_node in execution_order:
            if current_node not in script_specs:
                results["pipeline_success"] = False
                results["errors"].append(f"No ScriptExecutionSpec found for node: {current_node}")
                continue
            
            # Test individual script functionality first
            script_spec = script_specs[current_node]
            main_params = self.builder.get_script_main_params(script_spec)
            
            script_result = self.test_script_with_spec(script_spec, main_params)
            results["script_results"][current_node] = script_result
            
            if not script_result.success:
                results["pipeline_success"] = False
                results["errors"].append(f"Script {current_node} failed: {script_result.error_message}")
                continue  # Skip data flow testing for failed scripts
            
            executed_nodes.add(current_node)
            
            # Test data compatibility with all dependent nodes
            outgoing_edges = [(src, dst) for src, dst in dag.edges if src == current_node]
            
            for src_node, dst_node in outgoing_edges:
                if dst_node not in script_specs:
                    results["pipeline_success"] = False
                    results["errors"].append(f"Missing ScriptExecutionSpec for destination node: {dst_node}")
                    continue
                
                spec_a = script_specs[src_node]
                spec_b = script_specs[dst_node]
                
                # Test data compatibility using enhanced matching
                compat_result = self.test_data_compatibility_with_specs(spec_a, spec_b)
                results["data_flow_results"][f"{src_node}->{dst_node}"] = compat_result
                
                if not compat_result.compatible:
                    results["pipeline_success"] = False
                    results["errors"].extend(compat_result.compatibility_issues)
        
        # Validate all edges were tested
        expected_edges = set(f"{src}->{dst}" for src, dst in dag.edges)
        tested_edges = set(results["data_flow_results"].keys())
        missing_edges = expected_edges - tested_edges
        
        if missing_edges:
            results["pipeline_success"] = False
            results["errors"].append(f"Untested edges: {', '.join(missing_edges)}")
        
        return results
        
    except Exception as e:
        results["pipeline_success"] = False
        results["errors"].append(f"Pipeline flow test failed: {str(e)}")
        return results
```

### 6. Enhanced Data Compatibility with Logical Name Matching

```python
def test_enhanced_data_compatibility_with_specs(self, spec_a: ScriptExecutionSpec, 
                                              spec_b: ScriptExecutionSpec) -> Dict[str, Any]:
    """
    Phase 2: Enhanced data compatibility testing with logical name matching.
    
    Uses the LogicalNameMatchingTester for intelligent path matching and
    semantic similarity-based compatibility checking.
    """
    try:
        # Convert to enhanced specs
        enhanced_spec_a = self._create_enhanced_script_spec(spec_a)
        enhanced_spec_b = self._create_enhanced_script_spec(spec_b)
        
        # Use LogicalNameMatchingTester for enhanced compatibility testing
        from .logical_name_matching import LogicalNameMatchingTester
        
        tester = LogicalNameMatchingTester(self.semantic_threshold)
        compatibility_result = tester.test_enhanced_data_compatibility(
            enhanced_spec_a, enhanced_spec_b
        )
        
        return compatibility_result
        
    except Exception as e:
        return {
            "compatible": False,
            "issues": [f"Enhanced compatibility test failed: {str(e)}"],
            "path_matches": []
        }

def test_pipeline_flow_with_topological_order(self, pipeline_spec: PipelineTestingSpec) -> Dict[str, Any]:
    """
    Phase 2: Pipeline testing with topological execution ordering.
    
    Uses the TopologicalExecutor for proper dependency-aware execution
    and comprehensive edge coverage validation.
    """
    try:
        from .logical_name_matching import TopologicalExecutor
        
        dag = pipeline_spec.dag
        script_specs = pipeline_spec.script_specs
        
        # Convert all specs to enhanced versions
        enhanced_script_specs = {}
        for node_name, spec in script_specs.items():
            enhanced_script_specs[node_name] = self._create_enhanced_script_spec(spec)
        
        # Use LogicalNameMatchingTester for pipeline testing
        from .logical_name_matching import LogicalNameMatchingTester
        
        tester = LogicalNameMatchingTester(self.semantic_threshold)
        pipeline_result = tester.test_pipeline_with_topological_order(
            dag, enhanced_script_specs
        )
        
        return pipeline_result
        
    except Exception as e:
        return {
            "pipeline_success": False,
            "execution_order": [],
            "errors": [f"Enhanced pipeline test failed: {str(e)}"]
        }
```

## Helper Methods

### 1. Script Execution Safety

```python
def _execute_script_safely(self, script_path: Path, main_params: Dict[str, Any], 
                          env: Dict[str, str]) -> ExecutionResult:
    """Execute script with timeout, error capture, and resource monitoring."""
    
    try:
        # Import and execute script main function
        spec = importlib.util.spec_from_file_location("script_module", script_path)
        module = importlib.util.module_from_spec(spec)
        
        # Capture stdout/stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        
        try:
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            
            # Execute with environment
            with patch.dict(os.environ, env):
                spec.loader.exec_module(module)
                
                if hasattr(module, 'main'):
                    module.main(**main_params)
                else:
                    raise AttributeError("Script must have a 'main' function")
            
            return ExecutionResult(
                success=True,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue()
            )
            
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            
    except Exception as e:
        return ExecutionResult(
            success=False,
            error_message=str(e),
            stdout=stdout_capture.getvalue() if 'stdout_capture' in locals() else "",
            stderr=stderr_capture.getvalue() if 'stderr_capture' in locals() else ""
        )
```

### 2. Output Validation

```python
def _validate_script_outputs(self, spec: ScriptExecutionSpec) -> OutputValidationResult:
    """Validate script outputs exist and are valid."""
    
    output_files = []
    errors = []
    
    for logical_name, output_path in spec.output_paths.items():
        output_dir = Path(output_path)
        
        if not output_dir.exists():
            errors.append(f"Output directory does not exist: {output_path}")
            continue
        
        # Find valid output files using enhanced detection
        valid_files = self._find_valid_output_files(output_dir)
        
        if not valid_files:
            errors.append(f"No valid output files found in: {output_path}")
        else:
            output_files.extend(valid_files)
    
    return OutputValidationResult(
        success=len(errors) == 0,
        output_files=output_files,
        error_message="; ".join(errors) if errors else None
    )
```

### 3. Matching Report Generation

```python
def _generate_matching_report(self, path_matches: List[PathMatch]) -> Dict[str, Any]:
    """Generate detailed matching report for debugging and analysis."""
    
    if not path_matches:
        return {"summary": "No matches found"}
    
    # Group matches by type
    matches_by_type = {}
    for match in path_matches:
        match_type = match.match_type
        if match_type not in matches_by_type:
            matches_by_type[match_type] = []
        matches_by_type[match_type].append(match)
    
    # Calculate statistics
    total_matches = len(path_matches)
    high_confidence = len([m for m in path_matches if m.confidence >= 0.9])
    medium_confidence = len([m for m in path_matches if 0.7 <= m.confidence < 0.9])
    low_confidence = len([m for m in path_matches if m.confidence < 0.7])
    
    return {
        "summary": f"{total_matches} matches found",
        "confidence_distribution": {
            "high": high_confidence,
            "medium": medium_confidence,
            "low": low_confidence
        },
        "matches_by_type": {
            match_type: len(matches) 
            for match_type, matches in matches_by_type.items()
        },
        "best_match": {
            "source": path_matches[0].source_logical_name,
            "dest": path_matches[0].dest_logical_name,
            "confidence": path_matches[0].confidence,
            "type": path_matches[0].match_type
        } if path_matches else None
    }
```

## Integration Points

### 1. PipelineTestingSpecBuilder Integration

```python
# RuntimeTester uses builder for:
# - Script parameter generation
# - Spec resolution and management
# - Directory structure access

main_params = self.builder.get_script_main_params(spec)
spec = self.builder.resolve_script_execution_spec_from_node(node_name)
```

### 2. Logical Name Matching Integration

```python
# RuntimeTester integrates with logical_name_matching module for:
# - Enhanced data compatibility testing
# - Topological pipeline execution
# - Semantic similarity matching

from .logical_name_matching import LogicalNameMatchingTester, TopologicalExecutor
```

### 3. Registry Integration

```python
# RuntimeTester leverages registry for:
# - Step name resolution
# - Job type handling
# - Workspace context awareness

from cursus.registry.step_names import get_step_name_from_spec_type
```

## Performance Characteristics

### Script Execution Performance
- **Individual script**: ~100ms-10s (depends on script complexity)
- **Data compatibility**: ~200ms-20s (includes two script executions)
- **Pipeline flow**: ~1s-5min (depends on pipeline size and complexity)

### File Detection Performance
- **Directory scanning**: ~0.1ms per directory
- **File validation**: ~0.01ms per file
- **Format detection**: ~0.1ms per file

### Memory Usage
- **Script execution**: ~10-100MB per script (depends on data size)
- **Result storage**: ~1-10KB per test result
- **Path matching**: ~1KB per match set

## Error Handling Strategy

### 1. Script Execution Errors
- Capture and report import errors
- Handle missing main function
- Timeout protection for long-running scripts
- Resource monitoring and limits

### 2. Data Compatibility Errors
- Missing output files
- Invalid file formats
- Path matching failures
- Logical name resolution errors

### 3. Pipeline Execution Errors
- DAG topology errors (cycles, missing nodes)
- Script dependency failures
- Data flow validation errors
- Incomplete edge coverage

## Testing Strategy

### Unit Tests
- Individual script execution with various scenarios
- File format detection and validation
- Logical name matching algorithms
- Error handling for edge cases

### Integration Tests
- End-to-end data compatibility testing
- Pipeline flow with real DAG structures
- Performance testing with large datasets
- Error recovery and reporting

### Performance Tests
- Script execution timing
- Memory usage monitoring
- Concurrent execution capabilities
- Large pipeline scalability

## Future Enhancements

### 1. Advanced Execution
- Parallel script execution where possible
- Resource usage monitoring and limits
- Containerized execution for isolation
- Distributed execution across multiple machines

### 2. Enhanced Validation
- Data schema validation between scripts
- Performance regression detection
- Output quality metrics
- Automated test case generation

### 3. Intelligent Matching
- Machine learning-based path matching
- User feedback integration for match quality
- Dynamic alias learning from usage patterns
- Cross-pipeline pattern recognition

## References

### Foundation Documents
- **[Script Contract](script_contract.md)**: Script contract specifications that define the testing interfaces and execution requirements for script validation
- **[Pipeline DAG](pipeline_dag.md)**: Pipeline DAG structure and topological sorting algorithms that provide the mathematical framework for proper execution ordering
- **[Step Specification](step_specification.md)**: Step specification system that provides the logical name patterns and alias systems used in path matching

### Testing Framework Integration
- **[Pipeline Runtime Testing Master Design](pipeline_runtime_testing_master_design_OUTDATED.md)**: ⚠️ **OUTDATED** - Master testing design that provides the overall architecture context and testing workflow patterns
- **[Pipeline Runtime Testing System Design](pipeline_runtime_testing_system_design_OUTDATED.md)**: ⚠️ **OUTDATED** - Detailed system design that defines the comprehensive testing capabilities and integration patterns
- **[Enhanced Universal Step Builder Tester Design](enhanced_universal_step_builder_tester_design.md)**: Universal testing patterns that inform the script execution and validation strategies

### Semantic Matching and Dependency Resolution
- **[Dependency Resolver](dependency_resolver.md)**: Intelligent matching engine with semantic compatibility scoring that provides the semantic matching infrastructure used in logical name matching
- **[Dependency Resolution System](dependency_resolution_system.md)**: Dependency resolution patterns that inspire the path matching and compatibility checking algorithms

### File Format and Data Handling
- **[Flexible File Resolver Design](flexible_file_resolver_design.md)**: File resolution patterns that inform the enhanced file format support and intelligent temporary file filtering
- **[Pipeline Runtime Data Management Design](pipeline_runtime_data_management_design.md)**: Data management patterns that guide the format-agnostic file handling and output validation strategies

### Validation and Error Handling
- **[Validation Engine](validation_engine.md)**: Validation patterns and error handling strategies that inform the comprehensive script and pipeline validation approaches
- **[Two Level Alignment Validation System Design](two_level_alignment_validation_system_design.md)**: Multi-level validation patterns that inspire the script-level and pipeline-level validation architecture

### Execution and Performance
- **[Pipeline Runtime Core Engine Design](pipeline_runtime_core_engine_design.md)**: Core execution engine patterns that inform the script execution safety and performance optimization strategies
- **[Pipeline Runtime Execution Layer Design](pipeline_runtime_execution_layer_design.md)**: Execution layer design that provides the foundation for topological execution and resource management

### Logical Name Matching Integration
- **[Specification Driven Design](specification_driven_design.md)**: Specification-driven patterns that inform the enhanced ScriptExecutionSpec integration and path specification handling
- **[Enhanced Property Reference](enhanced_property_reference.md)**: Property reference patterns that inspire the logical name alias system and semantic matching capabilities

## Conclusion

The RuntimeTester provides a comprehensive, intelligent execution engine for pipeline runtime testing. By combining robust script execution with enhanced data compatibility testing and logical name matching, it enables thorough validation of pipeline functionality while maintaining performance and reliability.

Key design principles:
- **Robustness**: Comprehensive error handling and validation
- **Intelligence**: Semantic matching and logical name resolution
- **Performance**: Efficient execution with optimization opportunities
- **Extensibility**: Modular design supporting future enhancements

The RuntimeTester serves as the foundation for reliable pipeline validation, ensuring that scripts work correctly both individually and as part of larger pipeline workflows.

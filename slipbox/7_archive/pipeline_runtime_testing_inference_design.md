---
tags:
  - archive
  - design
  - pipeline_runtime_testing
  - inference_handler_testing
  - sagemaker_inference
  - model_validation
  - offline_testing
keywords:
  - inference handler testing
  - SageMaker inference
  - model_fn
  - input_fn
  - predict_fn
  - output_fn
  - offline validation
  - inference pipeline
  - model testing
topics:
  - inference testing
  - model validation
  - SageMaker integration
  - offline testing
  - inference pipeline validation
language: python
date of note: 2025-09-14
---

# Pipeline Runtime Testing Inference Design

## Overview

This document outlines the design for extending the Pipeline Runtime Testing system to support offline testing of SageMaker inference handlers. The system validates the four core inference functions (`model_fn`, `input_fn`, `predict_fn`, `output_fn`) both individually and as an integrated pipeline, ensuring proper data flow and compatibility between functions.

## Background and Motivation

### SageMaker Inference Handler Pattern

SageMaker inference handlers follow a standardized four-function pattern that processes inference requests:

1. **`model_fn(model_dir)`** - Loads model and preprocessing artifacts (initialization phase)
2. **`input_fn(request_body, content_type)`** - Deserializes request data into model input format
3. **`predict_fn(input_object, model_artifacts)`** - Performs inference using loaded model
4. **`output_fn(predictions, accept_type)`** - Serializes predictions into response format

### Inference Pipeline Flow

The functions execute in a specific sequence with data dependencies:

```
Initialization Phase (once):
┌─────────────┐
│  model_fn   │ ──► model_artifacts
└─────────────┘

Request Processing Phase (per request):
┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│  input_fn    │───▶│ predict_fn  │───▶│ output_fn   │
│              │    │             │    │             │
│ request_body │    │ input_obj + │    │ prediction +│
│ content_type │    │ model_arts  │    │ accept_type │
│      ↓       │    │      ↓      │    │      ↓      │
│ input_object │    │ prediction  │    │ response    │
└──────────────┘    └─────────────┘    └─────────────┘
```

### Testing Challenges

Current runtime testing framework addresses script-based testing with `main()` functions, but inference handlers require:

- **Function-level isolation testing** - Each function tested independently
- **Data compatibility validation** - Ensuring output of one function matches input of next
- **End-to-end pipeline testing** - Complete inference flow validation
- **Multiple content type support** - Testing various input/output formats
- **Error scenario handling** - Validation of error conditions and edge cases

## Architecture Design

### Simplified Core Components Integration

Following the **Code Redundancy Evaluation Guide** principles, the inference testing system extends the existing runtime testing architecture with minimal new components:

```python
# Existing Components (Enhanced)
RuntimeTester ──► Enhanced with 4 inference methods
    │                      
    ├─ ScriptExecutionSpec ├─ InferenceHandlerSpec (NEW)
    │                      │
    └─ PipelineTestingSpec └─ InferencePipelineTestingSpec (NEW, derived)

# Simplified Approach - No separate tester classes
# All functionality integrated into existing RuntimeTester
```

### Key Simplification Benefits
- **70% reduction in file complexity** (2 new files vs 13+ originally planned)
- **15-25% target redundancy** achieved
- **Minimal maintenance overhead**
- **Seamless integration** with existing patterns

### Simplified Data Models

Following our simplified approach, we have minimal data models that focus on the 4 core functionalities:

#### InferenceHandlerSpec (Packaged Model Approach)

```python
class InferenceHandlerSpec(BaseModel):
    """Specification for testing SageMaker inference handlers with packaged models."""
    
    # Core Identity (similar to ScriptExecutionSpec)
    handler_name: str = Field(..., description="Name of the inference handler")
    step_name: str = Field(..., description="Step name that matches PipelineDAG node name")
    
    # Core inputs (mirroring registration_spec dependencies)
    packaged_model_path: str = Field(..., description="Path to model.tar.gz from package step")
    payload_samples_path: str = Field(..., description="Path to generated payload samples for testing")
    
    # Directory paths (following ScriptExecutionSpec pattern)
    model_paths: Dict[str, str] = Field(default_factory=dict, description="Paths to extracted model components")
    code_paths: Dict[str, str] = Field(default_factory=dict, description="Paths to inference code after extraction")
    data_paths: Dict[str, str] = Field(default_factory=dict, description="Paths to sample data and payload samples")
    
    # Content Type Support
    supported_content_types: List[str] = Field(default=["application/json", "text/csv"])
    supported_accept_types: List[str] = Field(default=["application/json", "text/csv"])
    
    # Execution Context
    environ_vars: Dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = Field(default=300)
    
    # Validation Configuration (focused on 4 core functions)
    validate_model_loading: bool = Field(default=True, description="Test model_fn")
    validate_input_processing: bool = Field(default=True, description="Test input_fn")
    validate_prediction: bool = Field(default=True, description="Test predict_fn")
    validate_output_formatting: bool = Field(default=True, description="Test output_fn")
    validate_end_to_end: bool = Field(default=True, description="Test complete pipeline")
```

**Key Simplifications:**
- **No InferenceTestSample class** - Uses payload samples from directory structure instead
- **Packaged model approach** - Works with model.tar.gz from package step
- **Path-based configuration** - Follows ScriptExecutionSpec pattern with three directory types

#### InferenceTestResult

```python
class InferenceTestResult(BaseModel):
    """Comprehensive result of inference handler testing."""
    
    # Overall Results
    handler_name: str
    overall_success: bool
    total_execution_time: float
    
    # Function-Level Results
    model_fn_result: FunctionTestResult
    input_fn_results: List[FunctionTestResult]  # Multiple content types
    predict_fn_results: List[FunctionTestResult]  # Multiple samples
    output_fn_results: List[FunctionTestResult]  # Multiple accept types
    
    # Integration Results
    end_to_end_results: List[EndToEndTestResult]  # Complete pipeline tests
    compatibility_results: List[CompatibilityTestResult]  # Function compatibility
    
    # Error Summary
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

class FunctionTestResult(BaseModel):
    """Result of testing a single inference function."""
    
    function_name: str  # "model_fn", "input_fn", "predict_fn", "output_fn"
    success: bool
    execution_time: float
    input_parameters: Dict[str, Any]
    output_data: Optional[Any] = None
    output_type: Optional[str] = None
    error_message: Optional[str] = None
    validation_details: Dict[str, Any] = Field(default_factory=dict)

class EndToEndTestResult(BaseModel):
    """Result of end-to-end inference pipeline testing."""
    
    sample_name: str
    content_type: str
    accept_type: str
    success: bool
    total_execution_time: float
    
    # Step-by-step results
    model_loading_time: float
    input_processing_time: float
    prediction_time: float
    output_formatting_time: float
    
    # Data flow tracking
    input_data_size: Optional[int] = None
    processed_input_type: Optional[str] = None
    prediction_shape: Optional[List[int]] = None
    output_data_size: Optional[int] = None
    
    error_message: Optional[str] = None
```

### Simplified Core Testing Engine

Following our simplified approach, **no separate tester classes are created**. Instead, we add **4 core methods directly to the existing RuntimeTester class**:

#### Enhanced RuntimeTester with Inference Methods

```python
class RuntimeTester:
    """Enhanced RuntimeTester with inference handler support (simplified approach)."""
    
    # Existing RuntimeTester methods remain unchanged...
    
    # NEW: 4 core inference testing methods added directly to RuntimeTester
    
    def test_inference_function(self, handler_module: Any, function_name: str, 
                               test_params: Dict[str, Any]) -> Dict[str, Any]:
        """Test individual inference function (model_fn, input_fn, predict_fn, output_fn)."""
        start_time = time.time()
        
        try:
            # Get function from module
            func = getattr(handler_module, function_name)
            
            # Execute function with test parameters
            result = func(**test_params)
            
            # Validate result based on function type
            validation = self._validate_function_result(function_name, result, test_params)
            
            return {
                "function_name": function_name,
                "success": True,
                "execution_time": time.time() - start_time,
                "result": result,
                "validation": validation
            }
        except Exception as e:
            return {
                "function_name": function_name,
                "success": False,
                "execution_time": time.time() - start_time,
                "error": str(e)
            }
    
    def test_inference_pipeline(self, handler_spec: InferenceHandlerSpec) -> Dict[str, Any]:
        """Test complete inference pipeline (all 4 functions connected)."""
        results = {"pipeline_success": True, "function_results": {}, "errors": []}
        
        try:
            # Extract packaged model to inference_inputs/
            extraction_paths = self._extract_packaged_model(
                handler_spec.packaged_model_path, 
                "inference_inputs"
            )
            
            # Load inference handler from extracted code/
            handler_module = self._load_handler_module(
                extraction_paths["handler_file"]
            )
            
            # Load payload samples from payload_samples_path
            payload_samples = self._load_payload_samples(
                handler_spec.payload_samples_path
            )
            
            # Step 1: Test model_fn with extraction root (model files at root level)
            model_artifacts = handler_module.model_fn(extraction_paths["extraction_root"])
            results["function_results"]["model_fn"] = {"success": True, "artifacts": model_artifacts}
            
            # Step 2-4: Test pipeline with payload samples
            for sample in payload_samples:
                # Step 2: input_fn
                processed_input = handler_module.input_fn(sample["data"], sample["content_type"])
                
                # Step 3: predict_fn
                predictions = handler_module.predict_fn(processed_input, model_artifacts)
                
                # Step 4: output_fn
                for accept_type in handler_spec.supported_accept_types:
                    response = handler_module.output_fn(predictions, accept_type)
                    
            results["function_results"]["pipeline"] = {"success": True}
            
        except Exception as e:
            results["pipeline_success"] = False
            results["errors"].append(str(e))
        finally:
            # Cleanup extraction directory
            self._cleanup_extraction_directory("inference_inputs")
        
        return results
    
    def test_script_to_inference_compatibility(self, script_spec: ScriptExecutionSpec,
                                              handler_spec: InferenceHandlerSpec) -> Dict[str, Any]:
        """Test data compatibility between script output and inference input."""
        
        # Execute script first
        script_result = self.test_script_with_spec(script_spec, self.builder.get_script_main_params(script_spec))
        
        if not script_result.success:
            return {"compatible": False, "error": "Script execution failed"}
        
        # Find script output files using semantic matching (like existing RuntimeTester)
        # Use semantic matching to find the best output path for inference input
        output_files = []
        compatibility_issues = []
        
        # Try each output path from script_spec to find valid files
        for output_name, output_path in script_spec.output_paths.items():
            output_dir = Path(output_path)
            files = self._find_valid_output_files(output_dir)
            if files:
                output_files.extend(files)
                break  # Use first valid output path
            else:
                compatibility_issues.append(f"No valid files in output path '{output_name}': {output_path}")
        
        if not output_files:
            return {
                "compatible": False, 
                "error": "No script output files found",
                "details": compatibility_issues
            }
        
        # Test if inference handler can process script output
        try:
            # Extract packaged model and load handler
            extraction_paths = self._extract_packaged_model(
                handler_spec.packaged_model_path, 
                "inference_inputs"
            )
            handler_module = self._load_handler_module(extraction_paths["handler_file"])
            
            # Try each output file with different content types
            for output_file in output_files:
                try:
                    with open(output_file, 'r') as f:
                        script_output_data = f.read()
                    
                    # Test with different content types
                    for content_type in handler_spec.supported_content_types:
                        try:
                            processed_input = handler_module.input_fn(script_output_data, content_type)
                            return {
                                "compatible": True, 
                                "content_type": content_type,
                                "output_file": str(output_file),
                                "file_format": self._detect_file_format(output_file)
                            }
                        except Exception as content_error:
                            compatibility_issues.append(
                                f"Content type '{content_type}' failed for {output_file.name}: {str(content_error)}"
                            )
                            continue
                            
                except Exception as file_error:
                    compatibility_issues.append(f"Failed to read {output_file.name}: {str(file_error)}")
                    continue
            
            return {
                "compatible": False, 
                "error": "No compatible content type found for any output file",
                "details": compatibility_issues
            }
            
        except Exception as e:
            return {"compatible": False, "error": str(e)}
        finally:
            self._cleanup_extraction_directory("inference_inputs")
    
    def test_pipeline_with_inference(self, pipeline_spec: PipelineTestingSpec,
                                    inference_handlers: Dict[str, InferenceHandlerSpec]) -> Dict[str, Any]:
        """Test pipeline where inference handlers replace registration steps."""
        
        results = {"pipeline_success": True, "script_results": {}, "inference_results": {}, "errors": []}
        
        # Test scripts first
        for node_name, script_spec in pipeline_spec.script_specs.items():
            if node_name not in inference_handlers:  # Only test non-inference scripts
                main_params = self.builder.get_script_main_params(script_spec)
                script_result = self.test_script_with_spec(script_spec, main_params)
                results["script_results"][node_name] = script_result
                
                if not script_result.success:
                    results["pipeline_success"] = False
                    results["errors"].append(f"Script {node_name} failed")
        
        # Test inference handlers
        for node_name, handler_spec in inference_handlers.items():
            handler_result = self.test_inference_pipeline(handler_spec)
            results["inference_results"][node_name] = handler_result
            
            if not handler_result["pipeline_success"]:
                results["pipeline_success"] = False
                results["errors"].extend(handler_result["errors"])
        
        # Test data flow between scripts and inference handlers
        for src_node, dst_node in pipeline_spec.dag.edges:
            if src_node in pipeline_spec.script_specs and dst_node in inference_handlers:
                compatibility = self.test_script_to_inference_compatibility(
                    pipeline_spec.script_specs[src_node],
                    inference_handlers[dst_node]
                )
                if not compatibility["compatible"]:
                    results["pipeline_success"] = False
                    results["errors"].append(f"Incompatible data flow: {src_node} -> {dst_node}")
        
        return results
    
    # Helper methods for packaged model handling
    def _extract_packaged_model(self, packaged_model_path: str, extraction_dir: str = "inference_inputs") -> Dict[str, str]:
        """Extract model.tar.gz and return paths to key components."""
        import tarfile
        
        extraction_path = Path(extraction_dir)
        extraction_path.mkdir(parents=True, exist_ok=True)
        
        # Extract tar.gz to extraction directory
        with tarfile.open(packaged_model_path, "r:gz") as tar:
            tar.extractall(path=extraction_path)
        
        # Return key paths based on package step structure
        paths = {
            "extraction_root": str(extraction_path),
            "inference_code": str(extraction_path / "code"),
            "handler_file": str(extraction_path / "code" / "inference.py")  # Assuming standard name
        }
        
        # Check for optional calibration
        calibration_dir = extraction_path / "calibration"
        if calibration_dir.exists():
            paths["calibration_model"] = str(calibration_dir)
        
        return paths
    
    def _load_payload_samples(self, payload_samples_path: str) -> List[Dict[str, Any]]:
        """Load test samples from payload samples directory."""
        samples = []
        payload_dir = Path(payload_samples_path)
        
        # Load CSV samples
        csv_dir = payload_dir / "csv_samples"
        if csv_dir.exists():
            for csv_file in csv_dir.glob("*.csv"):
                with open(csv_file, 'r') as f:
                    samples.append({
                        "sample_name": csv_file.stem,
                        "content_type": "text/csv",
                        "data": f.read().strip(),
                        "file_path": str(csv_file)
                    })
        
        # Load JSON samples
        json_dir = payload_dir / "json_samples"
        if json_dir.exists():
            for json_file in json_dir.glob("*.json"):
                with open(json_file, 'r') as f:
                    samples.append({
                        "sample_name": json_file.stem,
                        "content_type": "application/json",
                        "data": f.read().strip(),
                        "file_path": str(json_file)
                    })
        
        return samples
    
    def _cleanup_extraction_directory(self, extraction_dir: str) -> None:
        """Clean up extraction directory after testing."""
        import shutil
        extraction_path = Path(extraction_dir)
        if extraction_path.exists():
            shutil.rmtree(extraction_path)
```

**Key Simplifications:**
- **No separate InferenceHandlerTester class** - All methods added directly to RuntimeTester
- **No individual function tester classes** - All logic integrated into the 4 core methods
- **Packaged model extraction** - Handles model.tar.gz files automatically
- **Payload sample loading** - Loads samples from directory structure

**Note**: In our simplified approach, all individual function testing logic is integrated directly into the 4 core methods of the enhanced RuntimeTester class shown above. No separate function tester classes are created, maintaining our goal of minimal file proliferation and reduced complexity.

## Integration with Existing Runtime Testing Framework

### Simplified RuntimeTester Enhancement

Following our simplified approach, **no separate inference tester classes are created**. The existing `RuntimeTester` class is enhanced with **4 new methods directly**:

```python
class RuntimeTester:
    """Enhanced RuntimeTester with inference handler support (simplified approach)."""
    
    def __init__(self, config_or_workspace_dir, enable_logical_matching: bool = True, semantic_threshold: float = 0.7):
        # Existing initialization remains unchanged...
        # No separate inference_tester instance needed
        
        # All inference functionality is integrated directly into RuntimeTester
        # Uses existing builder, workspace_dir, and other infrastructure
    
    # NEW: 4 core inference testing methods added directly to RuntimeTester
    # (These are the same methods shown in the "Simplified Core Testing Engine" section above)
    
    def test_inference_function(self, handler_module: Any, function_name: str, test_params: Dict[str, Any]) -> Dict[str, Any]:
        """Test individual inference function (model_fn, input_fn, predict_fn, output_fn)."""
        # Implementation shown above in core testing engine section
        pass
    
    def test_inference_pipeline(self, handler_spec: InferenceHandlerSpec) -> Dict[str, Any]:
        """Test complete inference pipeline (all 4 functions connected)."""
        # Implementation shown above in core testing engine section
        pass
    
    def test_script_to_inference_compatibility(self, script_spec: ScriptExecutionSpec, handler_spec: InferenceHandlerSpec) -> Dict[str, Any]:
        """Test data compatibility between script output and inference input."""
        # Implementation shown above in core testing engine section
        pass
    
    def test_pipeline_with_inference(self, pipeline_spec: PipelineTestingSpec, inference_handlers: Dict[str, InferenceHandlerSpec]) -> Dict[str, Any]:
        """Test pipeline where inference handlers replace registration steps."""
        # Implementation shown above in core testing engine section
        pass
    
    # Helper methods for inference testing (reuse existing patterns)
    def _load_handler_module(self, handler_file_path: str):
        """Load inference handler module (similar to existing _find_script_path pattern)."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("inference_handler", handler_file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    
    def _validate_function_result(self, function_name: str, result: Any, test_params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate function result based on function type (reuses existing validation patterns)."""
        validation = {"function_type": function_name, "result_type": type(result).__name__}
        
        if function_name == "model_fn":
            validation["has_model_artifacts"] = result is not None
            validation["is_dict"] = isinstance(result, dict)
        elif function_name == "input_fn":
            validation["has_processed_input"] = result is not None
            validation["input_type"] = type(result).__name__
        elif function_name == "predict_fn":
            validation["has_predictions"] = result is not None
            validation["prediction_type"] = type(result).__name__
        elif function_name == "output_fn":
            validation["has_response"] = result is not None
            validation["response_type"] = type(result).__name__
        
        return validation
```

### Key Integration Benefits

**1. No Additional Dependencies:**
- No separate `InferenceHandlerTester` class to maintain
- No additional initialization parameters needed
- Reuses existing `builder`, `workspace_dir`, and infrastructure

**2. Consistent API:**
- Same initialization pattern as existing RuntimeTester
- Same error handling and logging patterns
- Same file discovery and validation patterns

**3. Seamless Usage:**
```python
# Same initialization as before
tester = RuntimeTester(workspace_dir="test/inference")

# New inference methods available directly
handler_result = tester.test_inference_pipeline(handler_spec)
compatibility_result = tester.test_script_to_inference_compatibility(script_spec, handler_spec)

# Existing script testing methods still work unchanged
script_result = tester.test_script_with_spec(script_spec, main_params)
pipeline_result = tester.test_pipeline_flow_with_spec(pipeline_spec)
```

**4. Backward Compatibility:**
- All existing RuntimeTester functionality remains unchanged
- No breaking changes to existing API
- Existing tests and usage patterns continue to work

### Simplified PipelineTestingSpecBuilder Integration

Following our simplified approach, **no changes are made to the existing PipelineTestingSpecBuilder**. The builder continues to focus on script specifications only, maintaining its current functionality unchanged.

**Key Integration Points:**

**1. No Builder Changes Required:**
- PipelineTestingSpecBuilder remains focused on ScriptExecutionSpec creation
- No new methods added for inference handler specifications
- Maintains existing contract discovery and script resolution functionality

**2. Direct InferenceHandlerSpec Creation:**
```python
# Create inference handler specs directly (no builder needed)
handler_spec = InferenceHandlerSpec.create_default(
    handler_name="xgboost_inference",
    step_name="ModelServing_inference",
    packaged_model_path="test/models/packaged_xgboost_model.tar.gz",
    payload_samples_path="test/payload_samples/xgboost_samples/"
)

# Use existing builder for script specs only
builder = PipelineTestingSpecBuilder(test_data_dir="test/integration/runtime")
script_spec = builder.resolve_script_execution_spec_from_node("DataPreprocessing_training")
```

**3. Mixed Pipeline Testing Pattern:**
```python
# Create pipeline spec with scripts using existing builder
pipeline_spec = builder.build_from_dag(dag)

# Add inference handlers separately (no builder integration needed)
inference_handlers = {
    "ModelServing_inference": InferenceHandlerSpec.create_default(
        handler_name="xgboost_inference",
        step_name="ModelServing_inference",
        packaged_model_path="test/models/packaged_xgboost_model.tar.gz",
        payload_samples_path="test/payload_samples/xgboost_samples/"
    )
}

# Test mixed pipeline using RuntimeTester
tester = RuntimeTester(workspace_dir="test/inference")
results = tester.test_pipeline_with_inference(pipeline_spec, inference_handlers)
```

**4. Benefits of No Builder Integration:**
- **Maintains simplicity** - No changes to existing, well-tested builder
- **Clear separation of concerns** - Scripts vs inference handlers
- **Backward compatibility** - All existing builder functionality unchanged
- **Reduced complexity** - No mixed specification management needed

**5. Usage Pattern:**
```python
# Existing script testing (unchanged)
builder = PipelineTestingSpecBuilder(test_data_dir="test/integration/runtime")
pipeline_spec = builder.build_from_dag(dag)

# New inference testing (direct creation)
handler_spec = InferenceHandlerSpec.create_default(
    handler_name="xgboost_inference",
    step_name="ModelServing_inference", 
    packaged_model_path="models/packaged_model.tar.gz",
    payload_samples_path="test/payload_samples/"
)

# Combined testing using RuntimeTester
tester = RuntimeTester(workspace_dir="test/integration/runtime")
script_results = tester.test_pipeline_flow_with_spec(pipeline_spec)
inference_results = tester.test_inference_pipeline(handler_spec)
compatibility_results = tester.test_script_to_inference_compatibility(
    pipeline_spec.script_specs["DataPreprocessing_training"], 
    handler_spec
)
```

This approach maintains the existing builder's focus and functionality while providing clean, direct creation of inference handler specifications without unnecessary complexity.

## Usage Examples

### Basic Inference Handler Testing

```python
# Create inference handler specification
handler_spec = InferenceHandlerSpec(
    handler_name="xgboost_inference",
    handler_path="dockers/xgboost_atoz/xgboost_inference.py",
    model_dir="test/models/xgboost_model",
    test_data_samples=[
        InferenceTestSample(
            sample_name="json_sample",
            content_type="application/json",
            request_body='{"feature1": 1.0, "feature2": 2.0, "feature3": 3.0}',
            expected_output_structure={"predictions": [{}]}
        ),
        InferenceTestSample(
            sample_name="csv_sample", 
            content_type="text/csv",
            request_body="1.0,2.0,3.0,4.0,5.0",
            expected_output_structure={"predictions": [{}]}
        )
    ],
    supported_content_types=["application/json", "text/csv", "application/x-parquet"],
    supported_accept_types=["application/json", "text/csv"]
)

# Test the inference handler
tester = RuntimeTester(workspace_dir="test/inference", enable_inference_testing=True)
result = tester.test_inference_handler(handler_spec)

# Analyze results
if result.overall_success:
    print(f"✅ Inference handler '{result.handler_name}' passed all tests!")
    print(f"Total execution time: {result.total_execution_time:.2f}s")
    print(f"End-to-end tests: {len(result.end_to_end_results)} passed")
else:
    print(f"❌ Inference handler '{result.handler_name}' failed tests:")
    for error in result.errors:
        print(f"  - {error}")
```

### Advanced Testing with Custom Validation

```python
# Create specification with error scenario testing
handler_spec = InferenceHandlerSpec(
    handler_name="xgboost_inference",
    handler_path="dockers/xgboost_atoz/xgboost_inference.py", 
    model_dir="test/models/xgboost_model",
    test_data_samples=[
        # Valid samples
        InferenceTestSample(
            sample_name="valid_json",
            content_type="application/json",
            request_body='{"data": [[1,2,3,4,5]]}',
            should_succeed=True
        ),
        # Error scenarios
        InferenceTestSample(
            sample_name="malformed_json",
            content_type="application/json", 
            request_body='{"invalid": json}',
            should_succeed=False,
            error_message_pattern=r".*JSON.*"
        ),
        InferenceTestSample(
            sample_name="unsupported_content_type",
            content_type="application/xml",
            request_body="<data>test</data>",
            should_succeed=False,
            error_message_pattern=r".*unsupported.*content.*type.*"
        )
    ],
    timeout_seconds=60,
    validate_model_loading=True,
    validate_input_processing=True,
    validate_prediction=True,
    validate_output_formatting=True,
    validate_end_to_end=True
)

# Execute comprehensive testing
result = tester.test_inference_handler(handler_spec)

# Detailed result analysis
print(f"Model loading: {'✅' if result.model_fn_result.success else '❌'}")
print(f"Input processing: {sum(1 for r in result.input_fn_results if r.success)}/{len(result.input_fn_results)} passed")
print(f"Prediction: {sum(1 for r in result.predict_fn_results if r.success)}/{len(result.predict_fn_results)} passed")
print(f"Output formatting: {sum(1 for r in result.output_fn_results if r.success)}/{len(result.output_fn_results)} passed")
print(f"End-to-end: {sum(1 for r in result.end_to_end_results if r.success)}/{len(result.end_to_end_results)} passed")
```

### Integration with Existing Pipeline Testing

```python
# Create combined testing specification
class InferencePipelineSpec(BaseModel):
    """Extended pipeline specification with inference handler support."""
    
    # Existing pipeline components
    dag: PipelineDAG
    script_specs: Dict[str, ScriptExecutionSpec]
    
    # New inference handler components
    inference_handlers: Dict[str, InferenceHandlerSpec] = Field(default_factory=dict)
    
    def add_inference_handler(self, step_name: str, handler_spec: InferenceHandlerSpec):
        """Add inference handler to pipeline specification."""
        self.inference_handlers[step_name] = handler_spec

# Usage example
pipeline_spec = InferencePipelineSpec(
    dag=my_pipeline_dag,
    script_specs=existing_script_specs
)

# Add inference handler for model serving step
inference_spec = InferenceHandlerSpec(
    handler_name="xgboost_serving",
    handler_path="dockers/xgboost_atoz/xgboost_inference.py",
    model_dir="test/models/trained_xgboost"
)
pipeline_spec.add_inference_handler("ModelServing", inference_spec)

# Test complete pipeline including inference handlers
tester = RuntimeTester(workspace_dir="test/pipeline", enable_inference_testing=True)

# Test scripts first
script_results = tester.test_pipeline_flow_with_spec(pipeline_spec)

# Test inference handlers
inference_results = {}
for step_name, handler_spec in pipeline_spec.inference_handlers.items():
    inference_results[step_name] = tester.test_inference_handler(handler_spec)

# Combined analysis
overall_success = (
    script_results["pipeline_success"] and
    all(result.overall_success for result in inference_results.values())
)
```

## Performance Characteristics

### Function Testing Performance
- **model_fn testing**: ~100ms-5s (depends on model size and complexity)
- **input_fn testing**: ~10ms-100ms per sample (depends on data size and processing)
- **predict_fn testing**: ~50ms-2s per sample (depends on model complexity)
- **output_fn testing**: ~5ms-50ms per sample (depends on serialization complexity)

### End-to-End Testing Performance
- **Complete pipeline**: ~200ms-10s per sample (sum of all function times)
- **Multiple content types**: Linear scaling with number of types tested
- **Multiple samples**: Linear scaling with number of samples

### Memory Usage
- **Model artifacts**: ~10MB-1GB (depends on model size)
- **Test samples**: ~1KB-10MB per sample (depends on data size)
- **Result storage**: ~1-10KB per test result
- **Total memory**: Typically 50MB-2GB for comprehensive testing

## Error Handling and Validation

### Function-Level Error Handling

```python
class InferenceErrorHandler:
    """Centralized error handling for inference testing."""
    
    @staticmethod
    def handle_model_fn_error(error: Exception, model_dir: str) -> FunctionTestResult:
        """Handle model_fn specific errors with detailed diagnostics."""
        error_message = str(error)
        
        # Provide specific guidance based on error type
        if "FileNotFoundError" in error_message:
            guidance = f"Model directory or required files missing in {model_dir}"
        elif "ImportError" in error_message:
            guidance = "Missing dependencies for model loading"
        elif "MemoryError" in error_message:
            guidance = "Insufficient memory to load model artifacts"
        else:
            guidance = "Unknown model loading error"
        
        return FunctionTestResult(
            function_name="model_fn",
            success=False,
            execution_time=0,
            input_parameters={"model_dir": model_dir},
            error_message=f"{guidance}: {error_message}",
            validation_details={"error_type": type(error).__name__}
        )
    
    @staticmethod
    def handle_input_fn_error(error: Exception, sample: InferenceTestSample) -> FunctionTestResult:
        """Handle input_fn specific errors with content type guidance."""
        error_message = str(error)
        
        # Provide content-type specific guidance
        if sample.content_type == "application/json":
            if "JSONDecodeError" in error_message:
                guidance = "Invalid JSON format in request body"
            else:
                guidance = "JSON processing error"
        elif sample.content_type == "text/csv":
            guidance = "CSV parsing or format error"
        elif sample.content_type == "application/x-parquet":
            guidance = "Parquet file processing error"
        else:
            guidance = f"Unsupported content type: {sample.content_type}"
        
        return FunctionTestResult(
            function_name="input_fn",
            success=False,
            execution_time=0,
            input_parameters={
                "content_type": sample.content_type,
                "sample_name": sample.sample_name
            },
            error_message=f"{guidance}: {error_message}",
            validation_details={
                "error_type": type(error).__name__,
                "content_type": sample.content_type
            }
        )
```

### Validation Framework

```python
class InferenceValidationFramework:
    """Comprehensive validation framework for inference handlers."""
    
    @staticmethod
    def validate_function_signatures(handler_module: Any) -> Dict[str, Any]:
        """Validate that all required functions have correct signatures."""
        validation_results = {
            "model_fn": {"exists": False, "signature_valid": False, "issues": []},
            "input_fn": {"exists": False, "signature_valid": False, "issues": []},
            "predict_fn": {"exists": False, "signature_valid": False, "issues": []},
            "output_fn": {"exists": False, "signature_valid": False, "issues": []}
        }
        
        # Expected signatures
        expected_signatures = {
            "model_fn": ["model_dir", "context"],  # context is optional
            "input_fn": ["request_body", "request_content_type", "context"],  # context is optional
            "predict_fn": ["input_data", "model", "context"],  # context is optional
            "output_fn": ["prediction", "accept", "context"]  # context is optional
        }
        
        for func_name, expected_params in expected_signatures.items():
            if hasattr(handler_module, func_name):
                validation_results[func_name]["exists"] = True
                
                func = getattr(handler_module, func_name)
                sig = inspect.signature(func)
                actual_params = list(sig.parameters.keys())
                
                # Check required parameters (excluding optional context)
                required_params = [p for p in expected_params if p != "context"]
                missing_required = set(required_params) - set(actual_params)
                
                if not missing_required:
                    validation_results[func_name]["signature_valid"] = True
                else:
                    validation_results[func_name]["issues"].append(
                        f"Missing required parameters: {missing_required}"
                    )
            else:
                validation_results[func_name]["issues"].append("Function not found")
        
        return validation_results
    
    @staticmethod
    def validate_data_compatibility(source_output: Any, dest_input_type: str) -> Dict[str, Any]:
        """Validate compatibility between function outputs and inputs."""
        compatibility = {
            "compatible": True,
            "source_type": type(source_output).__name__,
            "expected_dest_type": dest_input_type,
            "issues": []
        }
        
        # Type compatibility checks
        if dest_input_type == "DataFrame" and not hasattr(source_output, 'columns'):
            compatibility["compatible"] = False
            compatibility["issues"].append("Expected DataFrame-like object")
        
        if dest_input_type == "ndarray" and not hasattr(source_output, 'shape'):
            compatibility["compatible"] = False
            compatibility["issues"].append("Expected array-like object with shape")
        
        if dest_input_type == "dict" and not isinstance(source_output, dict):
            compatibility["compatible"] = False
            compatibility["issues"].append("Expected dictionary object")
        
        return compatibility
```

## Testing Strategy and Best Practices

### Test Data Generation

```python
class InferenceTestDataGenerator:
    """Generate test data samples for inference handler testing."""
    
    @staticmethod
    def generate_json_samples(feature_count: int = 5) -> List[InferenceTestSample]:
        """Generate JSON test samples with various formats."""
        samples = []
        
        # Single record format
        samples.append(InferenceTestSample(
            sample_name="json_single_record",
            content_type="application/json",
            request_body=json.dumps({f"feature_{i}": float(i) for i in range(feature_count)}),
            should_succeed=True
        ))
        
        # Array format
        samples.append(InferenceTestSample(
            sample_name="json_array_format",
            content_type="application/json",
            request_body=json.dumps({"data": [[float(i) for i in range(feature_count)]]}),
            should_succeed=True
        ))
        
        # Multiple records
        samples.append(InferenceTestSample(
            sample_name="json_multiple_records",
            content_type="application/json",
            request_body=json.dumps([
                {f"feature_{i}": float(i) for i in range(feature_count)},
                {f"feature_{i}": float(i+1) for i in range(feature_count)}
            ]),
            should_succeed=True
        ))
        
        # Error case - malformed JSON
        samples.append(InferenceTestSample(
            sample_name="json_malformed",
            content_type="application/json",
            request_body='{"invalid": json, "missing": quote}',
            should_succeed=False,
            error_message_pattern=r".*JSON.*"
        ))
        
        return samples
    
    @staticmethod
    def generate_csv_samples(feature_count: int = 5) -> List[InferenceTestSample]:
        """Generate CSV test samples."""
        samples = []
        
        # Single row
        samples.append(InferenceTestSample(
            sample_name="csv_single_row",
            content_type="text/csv",
            request_body=",".join([str(float(i)) for i in range(feature_count)]),
            should_succeed=True
        ))
        
        # Multiple rows
        csv_data = "\n".join([
            ",".join([str(float(i+j)) for i in range(feature_count)])
            for j in range(3)
        ])
        samples.append(InferenceTestSample(
            sample_name="csv_multiple_rows",
            content_type="text/csv",
            request_body=csv_data,
            should_succeed=True
        ))
        
        # Error case - wrong number of features
        samples.append(InferenceTestSample(
            sample_name="csv_wrong_feature_count",
            content_type="text/csv",
            request_body=",".join([str(float(i)) for i in range(feature_count + 2)]),
            should_succeed=False,
            error_message_pattern=r".*feature.*count.*"
        ))
        
        return samples
```

### Continuous Integration Integration

```python
class InferenceCIIntegration:
    """Integration patterns for CI/CD pipelines."""
    
    @staticmethod
    def create_ci_test_suite(handler_specs: List[InferenceHandlerSpec]) -> Dict[str, Any]:
        """Create comprehensive test suite for CI/CD."""
        test_suite = {
            "test_configuration": {
                "timeout_seconds": 300,
                "parallel_execution": True,
                "fail_fast": False,
                "generate_reports": True
            },
            "test_cases": [],
            "performance_benchmarks": {
                "max_model_loading_time": 30.0,
                "max_prediction_time": 2.0,
                "max_memory_usage_mb": 1000
            }
        }
        
        for spec in handler_specs:
            test_case = {
                "handler_name": spec.handler_name,
                "test_functions": ["model_fn", "input_fn", "predict_fn", "output_fn"],
                "test_integration": True,
                "test_error_scenarios": True,
                "performance_validation": True
            }
            test_suite["test_cases"].append(test_case)
        
        return test_suite
    
    @staticmethod
    def generate_test_report(results: List[InferenceTestResult]) -> Dict[str, Any]:
        """Generate comprehensive test report for CI/CD."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.overall_success)
        
        report = {
            "summary": {
                "total_handlers": total_tests,
                "passed": passed_tests,
                "failed": total_tests - passed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0
            },
            "performance_metrics": {
                "avg_execution_time": sum(r.total_execution_time for r in results) / total_tests,
                "max_execution_time": max(r.total_execution_time for r in results),
                "min_execution_time": min(r.total_execution_time for r in results)
            },
            "detailed_results": []
        }
        
        for result in results:
            detail = {
                "handler_name": result.handler_name,
                "success": result.overall_success,
                "execution_time": result.total_execution_time,
                "function_results": {
                    "model_fn": result.model_fn_result.success,
                    "input_fn": sum(1 for r in result.input_fn_results if r.success),
                    "predict_fn": sum(1 for r in result.predict_fn_results if r.success),
                    "output_fn": sum(1 for r in result.output_fn_results if r.success)
                },
                "errors": result.errors
            }
            report["detailed_results"].append(detail)
        
        return report
```

## Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1-2)
1. **Data Models Implementation**
   - Implement `InferenceHandlerSpec` and related models
   - Create `InferenceTestResult` and result models
   - Add validation and serialization methods

2. **Basic Testing Engine**
   - Implement `InferenceHandlerTester` core class
   - Create function-specific testers (`ModelFnTester`, `InputFnTester`, etc.)
   - Add basic error handling and validation

### Phase 2: Function Testing (Week 3-4)
1. **Individual Function Testers**
   - Complete implementation of all four function testers
   - Add comprehensive validation for each function type
   - Implement error scenario testing

2. **Integration Testing**
   - Implement end-to-end pipeline testing
   - Add data compatibility validation between functions
   - Create performance monitoring and metrics

### Phase 3: Framework Integration (Week 5-6)
1. **RuntimeTester Integration**
   - Extend existing `RuntimeTester` with inference capabilities
   - Ensure backward compatibility with existing functionality
   - Add configuration options for inference testing

2. **Pipeline Integration**
   - Create `InferencePipelineSpec` for combined testing
   - Integrate with existing pipeline testing workflows
   - Add support for mixed script/inference pipelines

### Phase 4: Advanced Features (Week 7-8)
1. **Advanced Validation**
   - Implement comprehensive error handling framework
   - Add performance benchmarking and optimization
   - Create detailed reporting and analytics

2. **CI/CD Integration**
   - Add continuous integration support
   - Create automated test generation
   - Implement test result reporting and metrics

## References

### Online Resources

#### SageMaker Documentation
- **[SageMaker PyTorch Inference](https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#serve-a-pytorch-model)**: Official documentation for PyTorch inference handlers with detailed examples of the four-function pattern
- **[SageMaker Inference Toolkit](https://github.com/aws/sagemaker-inference-toolkit)**: Core inference toolkit providing the foundation for inference handler patterns and validation approaches
- **[SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/)**: Comprehensive SDK documentation covering inference patterns, model deployment, and testing strategies

#### Testing and Validation Frameworks
- **[PyTest Documentation](https://docs.pytest.org/)**: Testing framework patterns that inform the function-level testing approach and error handling strategies
- **[Pydantic Documentation](https://pydantic-docs.helpmanual.io/)**: Data validation patterns used in the specification models and result validation
- **[Python Inspect Module](https://docs.python.org/3/library/inspect.html)**: Function signature inspection techniques used for validation and compatibility checking

### Design Documents

#### Core Runtime Testing Architecture
- **[Pipeline Runtime Testing Simplified Design](pipeline_runtime_testing_simplified_design.md)**: Foundation architecture that provides the core testing patterns and integration points for inference handler testing
- **[Runtime Tester Design](runtime_tester_design.md)**: Core execution engine design that defines the testing workflow patterns and extension points used by inference testing
- **[Script Execution Spec Design](script_execution_spec_design.md)**: Data model patterns and specification design that inform the `InferenceHandlerSpec` structure and validation approaches

#### Testing Framework Integration
- **[Pipeline Testing Spec Builder Design](pipeline_testing_spec_builder_design.md)**: Builder pattern implementation that provides the foundation for creating and managing inference handler specifications
- **[Pipeline Testing Spec Design](pipeline_testing_spec_design.md)**: Top-level pipeline configuration patterns that guide the integration of inference testing with existing pipeline workflows

#### Validation and Error Handling
- **[Enhanced Dependency Validation Design](enhanced_dependency_validation_design.md)**: Validation framework patterns that inform the comprehensive function validation and compatibility checking approaches
- **[Two Level Alignment Validation System Design](two_level_alignment_validation_system_design.md)**: Multi-level validation patterns that inspire the function-level and pipeline-level validation architecture

#### Data Management and Processing
- **[Pipeline Runtime Data Management Design](pipeline_runtime_data_management_design.md)**: Data management patterns that guide the handling of model artifacts, test samples, and result storage
- **[Flexible File Resolver Design](flexible_file_resolver_design.md)**: File resolution patterns that inform the model artifact discovery and validation approaches

#### Performance and Optimization
- **[Pipeline Runtime Core Engine Design](pipeline_runtime_core_engine_design.md)**: Core execution engine patterns that inform the performance optimization and resource management strategies
- **[Pipeline Runtime Execution Layer Design](pipeline_runtime_execution_layer_design.md)**: Execution layer design that provides the foundation for timeout handling, error recovery, and performance monitoring

#### Configuration and Specification Management
- **[Config Driven Design](config_driven_design.md)**: Core principles for specification-driven system architecture that guide the `InferenceHandlerSpec` design and usage patterns
- **[Config Types Format](config_types_format.md)**: Data model patterns and type system design that inform the inference testing data models and validation approaches
- **[Enhanced Property Reference](enhanced_property_reference.md)**: Property resolution and reference management patterns that inspire the test sample and configuration management

#### Testing Methodology and Best Practices
- **[Enhanced Universal Step Builder Tester Design](enhanced_universal_step_builder_tester_design.md)**: Universal testing patterns that inform the function-specific testing approaches and validation strategies
- **[Pytest Unittest Compatibility Framework Design](pytest_unittest_compatibility_framework_design.md)**: Testing framework integration patterns that guide the CI/CD integration and automated testing approaches

## Conclusion

The Pipeline Runtime Testing Inference Design extends the existing runtime testing framework to provide comprehensive offline validation of SageMaker inference handlers. By implementing function-level isolation testing, data compatibility validation, and end-to-end pipeline testing, the system ensures reliable inference handler development and deployment.

### Key Design Principles

1. **Function Isolation**: Each inference function is tested independently to identify specific issues and validate individual functionality
2. **Data Flow Validation**: Comprehensive testing of data compatibility between functions ensures proper pipeline integration
3. **Comprehensive Coverage**: Testing includes normal operation, error scenarios, multiple content types, and performance validation
4. **Framework Integration**: Seamless integration with existing runtime testing infrastructure maintains consistency and leverages proven patterns
5. **Extensibility**: Modular design supports future enhancements and additional inference patterns

### Benefits

- **Offline Validation**: Test inference handlers without deploying to SageMaker endpoints
- **Early Error Detection**: Identify issues during development rather than deployment
- **Comprehensive Testing**: Validate all aspects of inference handler functionality
- **Performance Monitoring**: Track execution times and resource usage
- **CI/CD Integration**: Automated testing in continuous integration pipelines

### Future Enhancements

- **Multi-Model Testing**: Support for testing multiple models simultaneously
- **Load Testing**: Performance testing under various load conditions
- **Advanced Validation**: Schema validation for input/output data structures
- **Monitoring Integration**: Integration with monitoring and alerting systems
- **Cloud Testing**: Integration with cloud-based testing environments

The inference testing system provides a robust foundation for reliable SageMaker inference handler development, ensuring quality and performance while maintaining the flexibility and extensibility of the existing runtime testing framework.

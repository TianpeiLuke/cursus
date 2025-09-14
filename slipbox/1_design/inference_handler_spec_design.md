---
tags:
  - design
  - data_model
  - inference_handler_testing
  - sagemaker_inference
  - specification
keywords:
  - InferenceHandlerSpec
  - inference handler specification
  - SageMaker inference
  - data model design
  - test configuration
  - validation framework
topics:
  - inference testing data models
  - specification design
  - validation framework
  - test configuration
language: python
date of note: 2025-09-14
---

# InferenceHandlerSpec Design

## Overview

The `InferenceHandlerSpec` is a core data model that defines the specification for testing SageMaker inference handlers. It encapsulates all necessary configuration for testing the four inference functions (`model_fn`, `input_fn`, `predict_fn`, `output_fn`) both individually and as an integrated pipeline.

## Design Principles

Following the **Code Redundancy Evaluation Guide** principles:
- **Extend existing patterns** from `ScriptExecutionSpec`
- **Minimal complexity** with essential fields only
- **Reuse validation patterns** from existing runtime testing models
- **Focus on 4 core functionalities** without over-engineering

## Data Model Definition

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class InferenceHandlerSpec(BaseModel):
    """Specification for testing SageMaker inference handlers with packaged models."""
    
    # Core Identity (similar to ScriptExecutionSpec)
    handler_name: str = Field(..., description="Name of the inference handler")
    step_name: str = Field(..., description="Step name that matches PipelineDAG node name")
    
    # Core inputs (mirroring registration_spec dependencies)
    packaged_model_path: str = Field(..., description="Path to model.tar.gz from package step")
    payload_samples_path: str = Field(..., description="Path to generated payload samples for testing")
    
    # Directory paths (following ScriptExecutionSpec pattern)
    model_paths: Dict[str, str] = Field(
        default_factory=dict, 
        description="Paths to extracted model components"
    )
    code_paths: Dict[str, str] = Field(
        default_factory=dict, 
        description="Paths to inference code after extraction"
    )
    data_paths: Dict[str, str] = Field(
        default_factory=dict, 
        description="Paths to sample data and payload samples"
    )
    
    # Content Type Support
    supported_content_types: List[str] = Field(
        default=["application/json", "text/csv"],
        description="Content types the handler should support"
    )
    supported_accept_types: List[str] = Field(
        default=["application/json", "text/csv"],
        description="Accept types the handler should support"
    )
    
    # Execution Context (similar to ScriptExecutionSpec)
    environ_vars: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    timeout_seconds: int = Field(default=300, description="Timeout for inference operations")
    
    # Validation Configuration (focused on 4 core functions)
    validate_model_loading: bool = Field(default=True, description="Test model_fn")
    validate_input_processing: bool = Field(default=True, description="Test input_fn")
    validate_prediction: bool = Field(default=True, description="Test predict_fn")
    validate_output_formatting: bool = Field(default=True, description="Test output_fn")
    validate_end_to_end: bool = Field(default=True, description="Test complete pipeline")
    
    # Metadata
    created_at: Optional[datetime] = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = Field(default_factory=datetime.now)
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def validate_configuration(self) -> List[str]:
        """Validate the specification configuration."""
        errors = []
        
        # Validate packaged model path
        from pathlib import Path
        packaged_model = Path(self.packaged_model_path)
        if not packaged_model.exists():
            errors.append(f"Packaged model file does not exist: {self.packaged_model_path}")
        elif not self.packaged_model_path.endswith('.tar.gz'):
            errors.append(f"Packaged model must be a .tar.gz file: {self.packaged_model_path}")
        
        # Validate payload samples path
        payload_samples = Path(self.payload_samples_path)
        if not payload_samples.exists():
            errors.append(f"Payload samples path does not exist: {self.payload_samples_path}")
        
        # Validate extracted paths (if set)
        if "extraction_root" in self.model_paths:
            extraction_root = Path(self.model_paths["extraction_root"])
            if extraction_root.exists():
                # Check for expected structure after extraction
                code_dir = extraction_root / "code"
                if not code_dir.exists():
                    errors.append(f"Expected code directory not found: {code_dir}")
        
        return errors
    
    def is_valid(self) -> bool:
        """Check if the specification is valid."""
        return len(self.validate_configuration()) == 0
    
    # Convenience methods for path access (similar to ScriptExecutionSpec pattern)
    def get_packaged_model_path(self) -> str:
        """Get the packaged model.tar.gz file path."""
        return self.packaged_model_path
    
    def get_payload_samples_path(self) -> str:
        """Get the payload samples directory path."""
        return self.payload_samples_path
    
    def get_extraction_root_path(self) -> Optional[str]:
        """Get the extraction root directory path."""
        return self.model_paths.get("extraction_root")
    
    def get_inference_code_path(self) -> Optional[str]:
        """Get the inference code directory path."""
        return self.code_paths.get("inference_code_dir")
    
    def get_handler_file_path(self) -> Optional[str]:
        """Get the inference handler file path."""
        return self.code_paths.get("handler_file")
    
    @classmethod
    def create_default(
        cls,
        handler_name: str,
        step_name: str,
        packaged_model_path: str,
        payload_samples_path: str,
        test_data_dir: str = "test/integration/inference",
    ) -> "InferenceHandlerSpec":
        """Create a default InferenceHandlerSpec with minimal setup."""
        return cls(
            handler_name=handler_name,
            step_name=step_name,
            packaged_model_path=packaged_model_path,
            payload_samples_path=payload_samples_path,
            model_paths={"extraction_root": f"{test_data_dir}/inference_inputs"},
            code_paths={"inference_code_dir": f"{test_data_dir}/inference_inputs/code"},
            data_paths={"payload_samples": payload_samples_path},
            environ_vars={"INFERENCE_MODE": "testing"},
        )
```

## Payload Samples Structure

Based on the XGBoost inference handler analysis, the payload samples directory contains actual test data files in different content types that the inference handler supports.

### Supported Content Types
- **`text/csv`**: Headerless CSV with feature values
- **`application/json`**: JSON objects with feature data
- **`application/x-parquet`**: Parquet files with structured data

### Example Payload Samples Directory Structure
```
test/payload_samples/xgboost_samples/
├── csv_samples/
│   ├── sample_001.csv          # "1.0,2.0,3.0,4.0,5.0"
│   ├── sample_002.csv          # "0.5,1.5,2.5,3.5,4.5"
│   └── batch_sample.csv        # Multiple rows of CSV data
├── json_samples/
│   ├── single_record.json      # {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0}
│   ├── array_records.json      # [{"feature1": 1.0, "feature2": 2.0}, {"feature1": 0.5, "feature2": 1.5}]
│   └── ndjson_sample.json      # Multiple JSON objects separated by newlines
└── parquet_samples/
    ├── sample_data.parquet     # Structured parquet file with feature columns
    └── batch_data.parquet      # Batch parquet data
```

### Example Payload Sample Contents

**CSV Sample (`csv_samples/sample_001.csv`)**:
```csv
1.0,2.0,3.0,4.0,5.0
```

**JSON Single Record (`json_samples/single_record.json`)**:
```json
{
  "feature1": 1.0,
  "feature2": 2.0,
  "feature3": 3.0,
  "feature4": 4.0,
  "feature5": 5.0
}
```

**JSON Array Records (`json_samples/array_records.json`)**:
```json
[
  {"feature1": 1.0, "feature2": 2.0, "feature3": 3.0, "feature4": 4.0, "feature5": 5.0},
  {"feature1": 0.5, "feature2": 1.5, "feature3": 2.5, "feature4": 3.5, "feature5": 4.5}
]
```

**NDJSON Sample (`json_samples/ndjson_sample.json`)**:
```json
{"feature1": 1.0, "feature2": 2.0, "feature3": 3.0, "feature4": 4.0, "feature5": 5.0}
{"feature1": 0.5, "feature2": 1.5, "feature3": 2.5, "feature4": 3.5, "feature5": 4.5}
{"feature1": 2.0, "feature2": 3.0, "feature3": 4.0, "feature4": 5.0, "feature5": 6.0}
```

### RuntimeTester Payload Loading

The RuntimeTester will automatically discover and load these payload samples:

```python
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
    
    # Load Parquet samples
    parquet_dir = payload_dir / "parquet_samples"
    if parquet_dir.exists():
        for parquet_file in parquet_dir.glob("*.parquet"):
            with open(parquet_file, 'rb') as f:
                samples.append({
                    "sample_name": parquet_file.stem,
                    "content_type": "application/x-parquet",
                    "data": f.read(),
                    "file_path": str(parquet_file)
                })
    
    return samples
```

## Usage Examples

### Basic Specification Creation with Packaged Model

```python
# Create inference handler specification with packaged model
handler_spec = InferenceHandlerSpec(
    handler_name="xgboost_inference",
    step_name="ModelServing_inference",
    packaged_model_path="test/models/packaged_xgboost_model.tar.gz",
    payload_samples_path="test/payload_samples/xgboost_samples/"
)

# Validate configuration
if handler_spec.is_valid():
    print("✅ Specification is valid")
else:
    print("❌ Validation errors:")
    for error in handler_spec.validate_configuration():
        print(f"  - {error}")
```

### Advanced Configuration with Custom Paths

```python
# Create specification with advanced configuration
handler_spec = InferenceHandlerSpec(
    handler_name="pytorch_inference",
    step_name="PyTorchServing_inference",
    packaged_model_path="test/models/packaged_pytorch_model.tar.gz",
    payload_samples_path="test/payload_samples/pytorch_samples/",
    model_paths={
        "extraction_root": "test/inference_inputs/pytorch_model/",
        "calibration_model": "test/inference_inputs/pytorch_model/calibration/"
    },
    code_paths={
        "inference_code_dir": "test/inference_inputs/pytorch_model/code/",
        "handler_file": "test/inference_inputs/pytorch_model/code/inference.py"
    },
    data_paths={
        "payload_samples": "test/payload_samples/pytorch_samples/",
        "additional_test_data": "test/custom_samples/"
    },
    supported_content_types=["application/json", "text/csv", "application/x-parquet"],
    supported_accept_types=["application/json", "text/csv"],
    environ_vars={
        "MODEL_TYPE": "pytorch",
        "INFERENCE_MODE": "testing"
    },
    timeout_seconds=600,
    validate_model_loading=True,
    validate_input_processing=True,
    validate_prediction=True,
    validate_output_formatting=True,
    validate_end_to_end=True
)
```

### Using Default Creation Method

```python
# Create with default structure
handler_spec = InferenceHandlerSpec.create_default(
    handler_name="xgboost_inference",
    step_name="ModelServing_inference",
    packaged_model_path="test/models/packaged_xgboost_model.tar.gz",
    payload_samples_path="test/payload_samples/xgboost_samples/"
)

# Access paths using convenience methods
print(f"Packaged model: {handler_spec.get_packaged_model_path()}")
print(f"Payload samples: {handler_spec.get_payload_samples_path()}")
print(f"Extraction root: {handler_spec.get_extraction_root_path()}")
print(f"Handler file: {handler_spec.get_handler_file_path()}")
```

### Integration with RuntimeTester

```python
# Use with RuntimeTester
from cursus.validation.runtime import RuntimeTester

tester = RuntimeTester(workspace_dir="test/inference")

# Test complete pipeline (RuntimeTester handles extraction automatically)
pipeline_result = tester.test_inference_pipeline(handler_spec)

# Test script-to-inference compatibility
script_spec = ScriptExecutionSpec(
    script_name="data_preprocessing",
    step_name="DataPreprocessing_training",
    script_path="scripts/data_preprocessing.py",
    output_paths={"data_output": "test/output/processed_data"}
)

compatibility_result = tester.test_script_to_inference_compatibility(script_spec, handler_spec)
```

## Integration with Existing Framework

### Relationship to ScriptExecutionSpec

The `InferenceHandlerSpec` follows similar patterns to `ScriptExecutionSpec`:

```python
# Similar core identity fields
class ScriptExecutionSpec(BaseModel):
    script_name: str          # → handler_name
    script_path: str          # → handler_path
    environ_vars: Dict[str, str]  # → environ_vars (same)
    
# Extended for inference-specific needs
class InferenceHandlerSpec(BaseModel):
    handler_name: str         # Similar to script_name
    handler_path: str         # Similar to script_path
    model_dir: str           # New: model artifacts location
    test_data_samples: List[InferenceTestSample]  # New: test data
    supported_content_types: List[str]  # New: content type support
```

### Builder Integration

```python
# Integration with PipelineTestingSpecBuilder
class PipelineTestingSpecBuilder:
    
    def create_inference_handler_spec(self, handler_name: str, handler_path: str,
                                     model_dir: str, **kwargs) -> InferenceHandlerSpec:
        """Create InferenceHandlerSpec with default test samples."""
        
        # Generate default test samples
        default_samples = self._generate_default_test_samples(handler_path)
        
        return InferenceHandlerSpec(
            handler_name=handler_name,
            handler_path=handler_path,
            model_dir=model_dir,
            test_data_samples=kwargs.get('test_data_samples', default_samples),
            **kwargs
        )
```

## Validation and Error Handling

### Comprehensive Validation

```python
def validate_configuration(self) -> List[str]:
    """Comprehensive validation with specific error messages."""
    errors = []
    
    # Core field validation
    if not self.handler_name:
        errors.append("handler_name is required")
    
    if not self.handler_path:
        errors.append("handler_path is required")
    elif not self.handler_path.endswith('.py'):
        errors.append("handler_path must point to a Python file (.py)")
    
    # File system validation
    from pathlib import Path
    
    handler_file = Path(self.handler_path)
    if not handler_file.exists():
        errors.append(f"Handler file does not exist: {self.handler_path}")
    
    model_directory = Path(self.model_dir)
    if not model_directory.exists():
        errors.append(f"Model directory does not exist: {self.model_dir}")
    elif not model_directory.is_dir():
        errors.append(f"Model path is not a directory: {self.model_dir}")
    
    # Test sample validation
    if not self.test_data_samples:
        errors.append("At least one test sample is required")
    
    for i, sample in enumerate(self.test_data_samples):
        if sample.content_type not in self.supported_content_types:
            errors.append(f"Sample {i} content type '{sample.content_type}' not supported")
        
        if not sample.data:
            errors.append(f"Sample {i} data is empty")
    
    # Content type validation
    if not self.supported_content_types:
        errors.append("At least one supported content type is required")
    
    if not self.supported_accept_types:
        errors.append("At least one supported accept type is required")
    
    return errors
```

### Error Recovery

```python
def auto_fix_common_issues(self) -> List[str]:
    """Attempt to automatically fix common configuration issues."""
    fixes_applied = []
    
    # Add default test samples if none exist
    if not self.test_data_samples:
        self.add_test_sample(
            sample_name="default_json",
            content_type="application/json",
            data='{"data": [1, 2, 3, 4, 5]}'
        )
        fixes_applied.append("Added default JSON test sample")
    
    # Ensure basic content types are supported
    basic_types = ["application/json", "text/csv"]
    for content_type in basic_types:
        if content_type not in self.supported_content_types:
            self.supported_content_types.append(content_type)
            fixes_applied.append(f"Added support for {content_type}")
    
    # Update timestamp if fixes were applied
    if fixes_applied:
        self.updated_at = datetime.now()
    
    return fixes_applied
```

## Serialization and Persistence

### JSON Serialization

```python
# Save specification to file
def save_to_file(self, file_path: str) -> None:
    """Save specification to JSON file."""
    import json
    from pathlib import Path
    
    with open(file_path, 'w') as f:
        json.dump(self.dict(), f, indent=2, default=str)

# Load specification from file
@classmethod
def load_from_file(cls, file_path: str) -> 'InferenceHandlerSpec':
    """Load specification from JSON file."""
    import json
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return cls(**data)

# Get standard file name
def get_spec_file_name(self) -> str:
    """Get standard file name for this specification."""
    return f"{self.handler_name}_inference_spec.json"
```

### Usage with Persistence

```python
# Save and load specifications
handler_spec = InferenceHandlerSpec(
    handler_name="xgboost_inference",
    handler_path="dockers/xgboost_atoz/xgboost_inference.py",
    model_dir="test/models/xgboost_model"
)

# Save to file
spec_file = f"test/specs/{handler_spec.get_spec_file_name()}"
handler_spec.save_to_file(spec_file)

# Load from file
loaded_spec = InferenceHandlerSpec.load_from_file(spec_file)
```

## Performance Considerations

### Memory Usage
- **Basic spec**: ~1-2KB per instance
- **With test samples**: ~2-10KB depending on sample data size
- **JSON serialization**: ~1-5KB per file

### Validation Performance
- **Configuration validation**: ~1-5ms per spec
- **File system checks**: ~0.1ms per path
- **Test sample validation**: ~0.01ms per sample

## Testing Strategy

### Unit Tests

```python
def test_inference_handler_spec_creation():
    """Test basic specification creation."""
    spec = InferenceHandlerSpec(
        handler_name="test_handler",
        handler_path="test/handler.py",
        model_dir="test/models"
    )
    
    assert spec.handler_name == "test_handler"
    assert spec.handler_path == "test/handler.py"
    assert spec.model_dir == "test/models"
    assert len(spec.test_data_samples) == 0

def test_test_sample_management():
    """Test test sample addition and retrieval."""
    spec = InferenceHandlerSpec(
        handler_name="test_handler",
        handler_path="test/handler.py",
        model_dir="test/models"
    )
    
    # Add test sample
    spec.add_test_sample("json_test", "application/json", '{"test": true}')
    
    assert len(spec.test_data_samples) == 1
    assert spec.test_data_samples[0].sample_name == "json_test"
    
    # Filter by content type
    json

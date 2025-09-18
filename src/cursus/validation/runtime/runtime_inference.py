"""
Inference Handler Testing Data Models

Contains Pydantic models for SageMaker inference handler testing specifications and results.
Extends the runtime testing framework to support offline testing of inference handlers.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict, field_serializer
from datetime import datetime

# Import for inheritance
from .runtime_models import PipelineTestingSpec


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
    
    model_config = ConfigDict()

    # Custom serializer for datetime fields (Pydantic V2 approach)
    @field_serializer('created_at', 'updated_at')
    def serialize_datetime_fields(self, value: Optional[datetime]) -> Optional[str]:
        """Serialize datetime objects to ISO format strings"""
        if value is None:
            return None
        return value.isoformat()
    
    def validate_configuration(self) -> List[str]:
        """Validate the specification configuration."""
        errors = []
        
        # Validate packaged model path
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


class InferenceTestResult(BaseModel):
    """Comprehensive result of inference handler testing."""
    
    # Overall Results
    handler_name: str = Field(..., description="Name of the inference handler tested")
    overall_success: bool = Field(..., description="Whether all tests passed")
    total_execution_time: float = Field(..., description="Total time for all tests")
    test_timestamp: datetime = Field(default_factory=datetime.now, description="When the test was executed")
    
    # Function-Level Results (4 core functions)
    model_fn_result: Optional[Dict[str, Any]] = Field(None, description="Result of model_fn testing")
    input_fn_results: List[Dict[str, Any]] = Field(default_factory=list, description="Results of input_fn testing")
    predict_fn_results: List[Dict[str, Any]] = Field(default_factory=list, description="Results of predict_fn testing")
    output_fn_results: List[Dict[str, Any]] = Field(default_factory=list, description="Results of output_fn testing")
    
    # Integration Results
    end_to_end_results: List[Dict[str, Any]] = Field(default_factory=list, description="End-to-end pipeline test results")
    compatibility_results: List[Dict[str, Any]] = Field(default_factory=list, description="Compatibility test results")
    
    # Error Summary
    errors: List[str] = Field(default_factory=list, description="List of all errors encountered")
    warnings: List[str] = Field(default_factory=list, description="List of warnings")
    
    model_config = ConfigDict()

    # Custom serializer for datetime fields (Pydantic V2 approach)
    @field_serializer('test_timestamp')
    def serialize_datetime_fields(self, value: datetime) -> str:
        """Serialize datetime objects to ISO format strings"""
        return value.isoformat()
    
    def get_overall_success_rate(self) -> float:
        """Get overall success rate across all tests."""
        total_tests = 0
        successful_tests = 0
        
        # Count function tests
        if self.model_fn_result:
            total_tests += 1
            if self.model_fn_result.get("success", False):
                successful_tests += 1
        
        for results_list in [self.input_fn_results, self.predict_fn_results, self.output_fn_results]:
            total_tests += len(results_list)
            successful_tests += sum(1 for r in results_list if r.get("success", False))
        
        # Count end-to-end tests
        total_tests += len(self.end_to_end_results)
        successful_tests += sum(1 for r in self.end_to_end_results if r.get("success", False))
        
        # Count compatibility tests
        total_tests += len(self.compatibility_results)
        successful_tests += sum(1 for r in self.compatibility_results if r.get("compatible", False))
        
        return successful_tests / total_tests if total_tests > 0 else 0.0


class InferencePipelineTestingSpec(PipelineTestingSpec):
    """Extended pipeline specification supporting both scripts and inference handlers."""
    
    # Add inference handler support
    inference_handlers: Dict[str, InferenceHandlerSpec] = Field(
        default_factory=dict, 
        description="Inference handler specifications for pipeline steps"
    )
    
    def add_inference_handler(self, step_name: str, handler_spec: InferenceHandlerSpec) -> None:
        """Add inference handler to pipeline specification."""
        self.inference_handlers[step_name] = handler_spec
    
    def has_inference_handlers(self) -> bool:
        """Check if pipeline has any inference handlers."""
        return len(self.inference_handlers) > 0
    
    def get_inference_handler_names(self) -> List[str]:
        """Get list of inference handler step names."""
        return list(self.inference_handlers.keys())
    
    def get_mixed_step_types(self) -> Dict[str, str]:
        """Get mapping of step names to their types (script or inference)."""
        step_types = {}
        
        # Add script steps
        for step_name in self.script_specs.keys():
            step_types[step_name] = "script"
        
        # Add inference handler steps
        for step_name in self.inference_handlers.keys():
            step_types[step_name] = "inference"
        
        return step_types
    
    def validate_mixed_pipeline(self) -> List[str]:
        """Validate mixed pipeline configuration."""
        errors = []
        
        # Check for step name conflicts
        script_steps = set(self.script_specs.keys())
        inference_steps = set(self.inference_handlers.keys())
        conflicts = script_steps.intersection(inference_steps)
        
        if conflicts:
            errors.append(f"Step name conflicts between scripts and inference handlers: {conflicts}")
        
        # Validate all DAG nodes are covered
        dag_nodes = set(self.dag.nodes)
        covered_nodes = script_steps.union(inference_steps)
        missing_nodes = dag_nodes - covered_nodes
        
        if missing_nodes:
            errors.append(f"DAG nodes without specifications: {missing_nodes}")
        
        # Validate extra specifications
        extra_nodes = covered_nodes - dag_nodes
        if extra_nodes:
            errors.append(f"Specifications for non-existent DAG nodes: {extra_nodes}")
        
        return errors
    
    def is_valid_mixed_pipeline(self) -> bool:
        """Check if mixed pipeline configuration is valid."""
        return len(self.validate_mixed_pipeline()) == 0

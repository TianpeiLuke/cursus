"""S3 Output Path Registry for systematic tracking of S3 output paths across pipeline execution."""

from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict, Any, Optional, List

class S3OutputInfo(BaseModel):
    """Comprehensive S3 output information with metadata"""
    
    logical_name: str = Field(
        ...,
        description="Logical name of the output as defined in step specification"
    )
    s3_uri: str = Field(
        ...,
        description="Complete S3 URI where the output is stored"
    )
    property_path: str = Field(
        ...,
        description="SageMaker property path for runtime resolution"
    )
    data_type: str = Field(
        ...,
        description="Data type of the output (e.g., 'S3Uri', 'ModelArtifacts')"
    )
    step_name: str = Field(
        ...,
        description="Name of the step that produced this output"
    )
    job_type: Optional[str] = Field(
        None,
        description="Job type context (training, validation, testing, calibration)"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When this output was registered"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (container paths, output types, etc.)"
    )

class ExecutionMetadata(BaseModel):
    """Metadata about pipeline execution context"""
    
    pipeline_name: Optional[str] = None
    execution_id: Optional[str] = None
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_steps: int = 0
    completed_steps: int = 0
    
    def mark_step_completed(self) -> None:
        """Mark a step as completed"""
        self.completed_steps += 1
    
    def is_complete(self) -> bool:
        """Check if pipeline execution is complete"""
        return self.completed_steps >= self.total_steps

class S3OutputPathRegistry(BaseModel):
    """Centralized registry for tracking S3 output paths across pipeline execution"""
    
    step_outputs: Dict[str, Dict[str, S3OutputInfo]] = Field(
        default_factory=dict,
        description="Nested dict: step_name -> logical_name -> S3OutputInfo"
    )
    execution_metadata: ExecutionMetadata = Field(
        default_factory=ExecutionMetadata,
        description="Metadata about the pipeline execution"
    )
    
    def register_step_output(self, step_name: str, logical_name: str, output_info: S3OutputInfo) -> None:
        """Register an S3 output for a specific step
        
        Args:
            step_name: Name of the step that produced the output
            logical_name: Logical name of the output
            output_info: S3OutputInfo object with output details
        """
        if step_name not in self.step_outputs:
            self.step_outputs[step_name] = {}
        
        self.step_outputs[step_name][logical_name] = output_info
        self.execution_metadata.mark_step_completed()
    
    def get_step_output_info(self, step_name: str, logical_name: str) -> Optional[S3OutputInfo]:
        """Get S3 output information for a specific step and logical name
        
        Args:
            step_name: Name of the step
            logical_name: Logical name of the output
            
        Returns:
            S3OutputInfo object if found, None otherwise
        """
        return self.step_outputs.get(step_name, {}).get(logical_name)
    
    def get_step_output_path(self, step_name: str, logical_name: str) -> Optional[str]:
        """Get S3 URI for a specific step output
        
        Args:
            step_name: Name of the step
            logical_name: Logical name of the output
            
        Returns:
            S3 URI string if found, None otherwise
        """
        output_info = self.get_step_output_info(step_name, logical_name)
        return output_info.s3_uri if output_info else None
    
    def get_all_step_outputs(self, step_name: str) -> Dict[str, S3OutputInfo]:
        """Get all outputs for a specific step
        
        Args:
            step_name: Name of the step
            
        Returns:
            Dictionary mapping logical names to S3OutputInfo objects
        """
        return self.step_outputs.get(step_name, {})
    
    def list_all_steps(self) -> List[str]:
        """List all steps that have registered outputs
        
        Returns:
            List of step names
        """
        return list(self.step_outputs.keys())
    
    def get_outputs_by_data_type(self, data_type: str) -> List[S3OutputInfo]:
        """Get all outputs of a specific data type
        
        Args:
            data_type: Data type to filter by
            
        Returns:
            List of S3OutputInfo objects matching the data type
        """
        matching_outputs = []
        for step_outputs in self.step_outputs.values():
            for output_info in step_outputs.values():
                if output_info.data_type == data_type:
                    matching_outputs.append(output_info)
        return matching_outputs
    
    def get_outputs_by_job_type(self, job_type: str) -> List[S3OutputInfo]:
        """Get all outputs from a specific job type
        
        Args:
            job_type: Job type to filter by
            
        Returns:
            List of S3OutputInfo objects matching the job type
        """
        matching_outputs = []
        for step_outputs in self.step_outputs.values():
            for output_info in step_outputs.values():
                if output_info.job_type == job_type:
                    matching_outputs.append(output_info)
        return matching_outputs
    
    def resolve_property_path(self, property_path: str) -> Optional[str]:
        """Resolve a SageMaker property path to an S3 URI
        
        Args:
            property_path: SageMaker property path to resolve
            
        Returns:
            S3 URI if found, None otherwise
        """
        for step_outputs in self.step_outputs.values():
            for output_info in step_outputs.values():
                if output_info.property_path == property_path:
                    return output_info.s3_uri
        return None
    
    def create_registry_summary(self) -> Dict[str, Any]:
        """Create a summary of the registry contents
        
        Returns:
            Dictionary containing registry summary
        """
        total_outputs = sum(len(outputs) for outputs in self.step_outputs.values())
        
        data_types = set()
        job_types = set()
        
        for step_outputs in self.step_outputs.values():
            for output_info in step_outputs.values():
                data_types.add(output_info.data_type)
                if output_info.job_type:
                    job_types.add(output_info.job_type)
        
        return {
            "total_steps": len(self.step_outputs),
            "total_outputs": total_outputs,
            "data_types": list(data_types),
            "job_types": list(job_types),
            "execution_metadata": self.execution_metadata.model_dump(),
            "registry_created": datetime.now().isoformat()
        }
    
    def export_to_dict(self) -> Dict[str, Any]:
        """Export registry to dictionary format
        
        Returns:
            Dictionary representation of the registry
        """
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "S3OutputPathRegistry":
        """Create registry from dictionary data
        
        Args:
            data: Dictionary data to create registry from
            
        Returns:
            S3OutputPathRegistry instance
        """
        return cls.model_validate(data)
    
    def merge_registry(self, other_registry: "S3OutputPathRegistry") -> None:
        """Merge another registry into this one
        
        Args:
            other_registry: Another S3OutputPathRegistry to merge
        """
        for step_name, step_outputs in other_registry.step_outputs.items():
            if step_name not in self.step_outputs:
                self.step_outputs[step_name] = {}
            
            for logical_name, output_info in step_outputs.items():
                # Only merge if not already present (avoid overwriting)
                if logical_name not in self.step_outputs[step_name]:
                    self.step_outputs[step_name][logical_name] = output_info
        
        # Update execution metadata
        self.execution_metadata.total_steps = max(
            self.execution_metadata.total_steps,
            other_registry.execution_metadata.total_steps
        )
        self.execution_metadata.completed_steps = max(
            self.execution_metadata.completed_steps,
            other_registry.execution_metadata.completed_steps
        )

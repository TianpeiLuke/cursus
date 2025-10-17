"""
Script Test Result

Defines the result structure for script execution in the DAG-guided testing framework.
This mirrors the result classes in cursus/core but targets script testing results.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path


class ScriptTestResult(BaseModel):
    """
    Comprehensive result for script execution in DAG-guided testing.
    
    This class captures all relevant information about script execution,
    including success/failure status, execution time, outputs, and metadata.
    
    Attributes:
        script_name: Name of the executed script
        step_name: DAG node name for this script
        success: Whether the script executed successfully
        execution_time: Time taken to execute the script (seconds)
        error_message: Error message if execution failed
        has_main_function: Whether the script has a valid main() function
        output_files: List of output files created by the script
        warnings: List of warning messages
        metadata: Additional metadata about the execution
        framework_info: Framework-specific information (if detected)
        builder_consistency: Builder-script consistency information
    """
    
    # Core Result Fields
    script_name: str = Field(..., description="Name of the executed script")
    step_name: str = Field(..., description="DAG node name for this script")
    success: bool = Field(..., description="Whether script executed successfully")
    execution_time: float = Field(..., description="Execution time in seconds")
    
    # Error Information
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    has_main_function: bool = Field(default=True, description="Whether script has valid main()")
    
    # Output Information
    output_files: List[str] = Field(default_factory=list, description="List of created output files")
    warnings: Optional[List[str]] = Field(default=None, description="Warning messages")
    
    # Metadata and Enhancement Information
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional execution metadata")
    framework_info: Optional[Dict[str, Any]] = Field(default=None, description="Framework-specific information")
    builder_consistency: Optional[Dict[str, Any]] = Field(default=None, description="Builder consistency information")
    
    # Execution Context
    execution_timestamp: datetime = Field(default_factory=datetime.now, description="When the script was executed")
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Path: lambda v: str(v),
        }
    
    @classmethod
    def create_success(
        cls,
        script_name: str,
        step_name: str,
        execution_time: float,
        output_files: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> "ScriptTestResult":
        """
        Create a successful script test result.
        
        Args:
            script_name: Name of the executed script
            step_name: DAG node name
            execution_time: Time taken to execute
            output_files: List of output files created
            **kwargs: Additional metadata
            
        Returns:
            ScriptTestResult indicating success
        """
        return cls(
            script_name=script_name,
            step_name=step_name,
            success=True,
            execution_time=execution_time,
            output_files=output_files or [],
            metadata=kwargs if kwargs else None,
        )
    
    @classmethod
    def create_failure(
        cls,
        script_name: str,
        step_name: str,
        execution_time: float,
        error_message: str,
        has_main_function: bool = True,
        **kwargs: Any,
    ) -> "ScriptTestResult":
        """
        Create a failed script test result.
        
        Args:
            script_name: Name of the executed script
            step_name: DAG node name
            execution_time: Time taken before failure
            error_message: Description of the error
            has_main_function: Whether script has valid main()
            **kwargs: Additional metadata
            
        Returns:
            ScriptTestResult indicating failure
        """
        return cls(
            script_name=script_name,
            step_name=step_name,
            success=False,
            execution_time=execution_time,
            error_message=error_message,
            has_main_function=has_main_function,
            metadata=kwargs if kwargs else None,
        )
    
    def add_framework_info(self, framework: str, **info: Any) -> None:
        """
        Add framework-specific information to the result.
        
        Args:
            framework: Detected framework name
            **info: Additional framework-specific information
        """
        if self.framework_info is None:
            self.framework_info = {}
        
        self.framework_info.update({
            "detected_framework": framework,
            **info
        })
    
    def add_builder_consistency_info(self, consistent: bool, **details: Any) -> None:
        """
        Add builder-script consistency information.
        
        Args:
            consistent: Whether script is consistent with builder expectations
            **details: Additional consistency check details
        """
        if self.builder_consistency is None:
            self.builder_consistency = {}
        
        self.builder_consistency.update({
            "builder_consistent": consistent,
            **details
        })
    
    def add_warning(self, warning: str) -> None:
        """
        Add a warning message to the result.
        
        Args:
            warning: Warning message to add
        """
        if self.warnings is None:
            self.warnings = []
        self.warnings.append(warning)
    
    def get_output_summary(self) -> Dict[str, Any]:
        """
        Get a summary of script outputs.
        
        Returns:
            Dictionary with output summary information
        """
        return {
            "total_outputs": len(self.output_files),
            "output_files": self.output_files,
            "outputs_exist": [
                {"file": f, "exists": Path(f).exists()} 
                for f in self.output_files
            ],
            "missing_outputs": [
                f for f in self.output_files 
                if not Path(f).exists()
            ]
        }
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get a summary of script execution.
        
        Returns:
            Dictionary with execution summary
        """
        summary = {
            "script_name": self.script_name,
            "step_name": self.step_name,
            "success": self.success,
            "execution_time": self.execution_time,
            "execution_timestamp": self.execution_timestamp.isoformat(),
            "has_main_function": self.has_main_function,
        }
        
        if not self.success and self.error_message:
            summary["error_message"] = self.error_message
        
        if self.warnings:
            summary["warnings_count"] = len(self.warnings)
            summary["warnings"] = self.warnings
        
        if self.framework_info:
            summary["framework"] = self.framework_info.get("detected_framework", "Unknown")
        
        if self.builder_consistency:
            summary["builder_consistent"] = self.builder_consistency.get("builder_consistent", False)
        
        return summary
    
    def is_successful_with_outputs(self) -> bool:
        """
        Check if script was successful and produced expected outputs.
        
        Returns:
            True if successful and all output files exist
        """
        if not self.success:
            return False
        
        if not self.output_files:
            return True  # No outputs expected
        
        return all(Path(f).exists() for f in self.output_files)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the script execution.
        
        Returns:
            Dictionary with performance metrics
        """
        metrics = {
            "execution_time_seconds": self.execution_time,
            "execution_timestamp": self.execution_timestamp.isoformat(),
        }
        
        # Add framework-specific metrics if available
        if self.framework_info and "performance_metrics" in self.framework_info:
            metrics.update(self.framework_info["performance_metrics"])
        
        # Add metadata metrics if available
        if self.metadata and "performance" in self.metadata:
            metrics.update(self.metadata["performance"])
        
        return metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result to dictionary format.
        
        Returns:
            Dictionary representation of the result
        """
        return self.model_dump()
    
    def __str__(self) -> str:
        """String representation of the result."""
        status = "✅ SUCCESS" if self.success else "❌ FAILED"
        return f"ScriptTestResult({self.script_name}: {status} in {self.execution_time:.2f}s)"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"ScriptTestResult("
            f"script_name='{self.script_name}', "
            f"step_name='{self.step_name}', "
            f"success={self.success}, "
            f"execution_time={self.execution_time:.2f}s, "
            f"outputs={len(self.output_files)})"
        )

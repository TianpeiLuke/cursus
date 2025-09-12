"""
Data models for Pipeline Runtime Testing

Contains Pydantic models for runtime testing specifications and results.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING
from pydantic import BaseModel, Field
from datetime import datetime

# Import PipelineDAG for integration
from ...api.dag.base_dag import PipelineDAG

# Forward reference to avoid circular imports
if TYPE_CHECKING:
    from .logical_name_matching import EnhancedScriptExecutionSpec


class ScriptTestResult(BaseModel):
    """Simple result model for script testing"""

    script_name: str
    success: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0
    has_main_function: bool = False


class DataCompatibilityResult(BaseModel):
    """Result model for data compatibility testing"""

    script_a: str
    script_b: str
    compatible: bool
    compatibility_issues: List[str] = Field(default_factory=list)
    data_format_a: Optional[str] = None
    data_format_b: Optional[str] = None


class ScriptExecutionSpec(BaseModel):
    """User-owned specification for executing a single script with main() interface"""

    script_name: str = Field(..., description="Name of the script to test")
    step_name: str = Field(
        ..., description="Step name that matches PipelineDAG node name"
    )
    script_path: Optional[str] = Field(None, description="Full path to script file")

    # Main function parameters (exactly what script needs) - user-provided
    input_paths: Dict[str, str] = Field(
        default_factory=dict, description="Input paths for script main()"
    )
    output_paths: Dict[str, str] = Field(
        default_factory=dict, description="Output paths for script main()"
    )
    environ_vars: Dict[str, str] = Field(
        default_factory=dict, description="Environment variables for script main()"
    )
    job_args: Dict[str, Any] = Field(
        default_factory=dict, description="Job arguments for script main()"
    )

    # User metadata for reuse
    last_updated: Optional[str] = Field(
        None, description="Timestamp when spec was last updated"
    )
    user_notes: Optional[str] = Field(
        None, description="User notes about this script configuration"
    )

    def save_to_file(self, specs_dir: str) -> str:
        """Save ScriptExecutionSpec to JSON file for reuse with auto-generated filename"""
        # Update timestamp
        self.last_updated = datetime.now().isoformat()

        # Auto-generate filename based on script name for local runtime testing
        filename = f"{self.script_name}_runtime_test_spec.json"
        file_path = Path(specs_dir) / filename

        with open(file_path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

        return str(file_path)

    @classmethod
    def load_from_file(cls, script_name: str, specs_dir: str) -> "ScriptExecutionSpec":
        """Load ScriptExecutionSpec from JSON file using auto-generated filename"""
        # Auto-generate filename based on script name (same pattern as save_to_file)
        filename = f"{script_name}_runtime_test_spec.json"
        file_path = Path(specs_dir) / filename

        if not file_path.exists():
            raise FileNotFoundError(f"ScriptExecutionSpec file not found: {file_path}")

        with open(file_path, "r") as f:
            data = json.load(f)

        return cls(**data)

    @classmethod
    def create_default(
        cls,
        script_name: str,
        step_name: str,
        test_data_dir: str = "test/integration/runtime",
    ) -> "ScriptExecutionSpec":
        """Create a default ScriptExecutionSpec with minimal setup"""
        return cls(
            script_name=script_name,
            step_name=step_name,
            input_paths={"data_input": f"{test_data_dir}/{script_name}/input"},
            output_paths={"data_output": f"{test_data_dir}/{script_name}/output"},
            environ_vars={"LABEL_FIELD": "label"},
            job_args={"job_type": "testing"},
        )


class PipelineTestingSpec(BaseModel):
    """Specification for testing an entire pipeline flow"""

    model_config = {"arbitrary_types_allowed": True}

    # Copy of the pipeline DAG structure
    dag: PipelineDAG = Field(
        ...,
        description="Copy of Pipeline DAG defining step dependencies and execution order",
    )

    # Script execution specifications for each step (supports both basic and enhanced specs)
    script_specs: Dict[str, ScriptExecutionSpec] = Field(
        ...,
        description="Execution specs for each pipeline step (supports both ScriptExecutionSpec and EnhancedScriptExecutionSpec)",
    )

    # Testing workspace configuration
    test_workspace_root: str = Field(
        default="test/integration/runtime",
        description="Root directory for test data and outputs",
    )
    workspace_aware_root: Optional[str] = Field(
        None, description="Workspace-aware project root"
    )

    def has_enhanced_specs(self) -> bool:
        """Check if any script specs are enhanced specs"""
        from .logical_name_matching import EnhancedScriptExecutionSpec

        return any(
            isinstance(spec, EnhancedScriptExecutionSpec)
            for spec in self.script_specs.values()
        )

    def get_enhanced_specs(self) -> Dict[str, "EnhancedScriptExecutionSpec"]:
        """Get only the enhanced script specs"""
        from .logical_name_matching import EnhancedScriptExecutionSpec

        return {
            name: spec
            for name, spec in self.script_specs.items()
            if isinstance(spec, EnhancedScriptExecutionSpec)
        }

    def get_basic_specs(self) -> Dict[str, ScriptExecutionSpec]:
        """Get only the basic script specs"""
        from .logical_name_matching import EnhancedScriptExecutionSpec

        return {
            name: spec
            for name, spec in self.script_specs.items()
            if not isinstance(spec, EnhancedScriptExecutionSpec)
        }


class RuntimeTestingConfiguration(BaseModel):
    """Complete configuration for runtime testing system"""

    # Core pipeline testing specification
    pipeline_spec: PipelineTestingSpec = Field(
        ..., description="Pipeline testing specification"
    )

    # Testing mode configuration
    test_individual_scripts: bool = Field(
        default=True, description="Whether to test scripts individually first"
    )
    test_data_compatibility: bool = Field(
        default=True,
        description="Whether to test data compatibility between connected scripts",
    )
    test_pipeline_flow: bool = Field(
        default=True, description="Whether to test complete pipeline flow"
    )

    # Enhanced features configuration
    enable_enhanced_features: bool = Field(
        default=False,
        description="Whether to enable enhanced logical name matching features",
    )
    enable_logical_matching: bool = Field(
        default=False,
        description="Whether to enable logical name matching for data compatibility",
    )

    # Workspace configuration
    use_workspace_aware: bool = Field(
        default=False, description="Whether to use workspace-aware project structure"
    )

    def model_post_init(self, __context) -> None:
        """Post-initialization to auto-enable enhanced features if enhanced specs are present"""
        super().model_post_init(__context)

        # Auto-enable enhanced features if pipeline has enhanced specs
        if self.pipeline_spec.has_enhanced_specs():
            self.enable_enhanced_features = True
            self.enable_logical_matching = True

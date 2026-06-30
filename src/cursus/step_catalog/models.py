"""
Simple data models for the unified step catalog system.

Following the Code Redundancy Evaluation Guide principles, these models are
simple and focused, avoiding over-engineering while supporting all US1-US5 requirements.
"""

from pydantic import BaseModel, Field
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable


# A "builder provider" is anything the assembler can call with the fixed 5-kwarg signature
# (config, sagemaker_session, role, registry_manager, dependency_resolver) to get a StepBuilderBase
# instance. A per-step builder CLASS is already such a callable, so today the catalog returns classes
# and this alias is satisfied trivially (dual-mode). It exists so the catalog's resolution methods can
# later return non-class factories (the classless Design-B end-state, FZ 31e1d3g1) WITHOUT changing
# the caller contract: callers must treat the result as a provider callable, never assume a class.
# (Loose Callable[..., Any] rather than a Protocol — the assembler invokes it as a constructor and
# duck-types the result; there is intentionally no isinstance gate, see pipeline_assembler.py:190.)
BuilderProvider = Callable[..., Any]


class FileMetadata(BaseModel):
    """Simple metadata for component files."""

    path: Path = Field(..., description="Path to the component file")
    file_type: str = Field(
        ...,
        description="Type of component: 'script', 'contract', 'spec', 'builder', 'config'",
    )
    modified_time: datetime = Field(
        ..., description="Last modification time of the file"
    )

    model_config = {"arbitrary_types_allowed": True, "frozen": True}


class StepInfo(BaseModel):
    """Essential step information combining registry data with file metadata."""

    step_name: str = Field(..., description="Name of the step")
    workspace_id: str = Field(..., description="Workspace where the step is located")
    registry_data: Dict[str, Any] = Field(
        default_factory=dict, description="Data from cursus.registry.step_names"
    )
    file_components: Dict[str, Optional[FileMetadata]] = Field(
        default_factory=dict,
        description="Discovered file components keyed by component type",
    )

    model_config = {"arbitrary_types_allowed": True}

    @property
    def config_class(self) -> str:
        """Get config class name from registry data."""
        value = self.registry_data.get("config_class", "")
        return str(value) if value is not None else ""

    @property
    def sagemaker_step_type(self) -> str:
        """Get SageMaker step type from registry data."""
        value = self.registry_data.get("sagemaker_step_type", "")
        return str(value) if value is not None else ""

    @property
    def builder_step_name(self) -> str:
        """Get builder step name from registry data."""
        value = self.registry_data.get("builder_step_name", "")
        return str(value) if value is not None else ""

    @property
    def description(self) -> str:
        """Get step description from registry data."""
        value = self.registry_data.get("description", "")
        return str(value) if value is not None else ""


class StepSearchResult(BaseModel):
    """Simple search result for step queries."""

    step_name: str = Field(..., description="Name of the matching step")
    workspace_id: str = Field(..., description="Workspace where the step is located")
    match_score: float = Field(..., description="Relevance score (0.0 to 1.0)")
    match_reason: str = Field(
        ..., description="Reason for the match (e.g., 'name_match', 'fuzzy_match')"
    )
    components_available: List[str] = Field(
        default_factory=list,
        description="List of available component types for this step",
    )

    model_config = {"frozen": True}

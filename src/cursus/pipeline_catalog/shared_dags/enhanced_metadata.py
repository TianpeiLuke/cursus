"""
Enhanced DAGMetadata System with Zettelkasten Integration

This module implements the enhanced DAGMetadata system that integrates
Zettelkasten knowledge management principles with the existing pipeline
metadata infrastructure.
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum
import logging

# Import DAGMetadata from the parent module
from . import DAGMetadata

logger = logging.getLogger(__name__)


class ComplexityLevel(Enum):
    """Standardized complexity levels for pipelines."""

    SIMPLE = "simple"
    STANDARD = "standard"
    ADVANCED = "advanced"
    COMPREHENSIVE = "comprehensive"


class PipelineFramework(Enum):
    """Supported pipeline frameworks."""

    XGBOOST = "xgboost"
    PYTORCH = "pytorch"
    SKLEARN = "sklearn"
    GENERIC = "generic"
    FRAMEWORK_AGNOSTIC = "framework_agnostic"


class ZettelkastenMetadata(BaseModel):
    """
    Zettelkasten-specific metadata for pipelines.

    Implements the five core Zettelkasten principles:
    1. Atomicity - Each pipeline represents one atomic concept
    2. Connectivity - Explicit connections between pipelines
    3. Anti-categories - Tag-based emergent organization
    4. Manual linking - Curated connections over search
    5. Dual-form structure - Metadata separate from implementation
    """

    # Core identification (matches catalog_index.json structure)
    atomic_id: str
    title: str = ""
    single_responsibility: str

    # Atomicity metadata
    input_interface: List[str] = Field(default_factory=list)
    output_interface: List[str] = Field(default_factory=list)
    side_effects: str = "none"
    independence_level: str = "fully_self_contained"
    node_count: int = 1
    edge_count: int = 0

    # Core metadata (matches catalog structure)
    framework: str = ""  # Single framework string to match catalog
    complexity: str = ""  # Single complexity string to match catalog
    use_case: str = ""
    features: List[str] = Field(default_factory=list)
    mods_compatible: bool = False

    # File tracking
    source_file: str = ""
    migration_source: str = ""
    created_date: str = ""
    priority: str = "standard"

    # Connectivity metadata
    connection_types: List[str] = Field(
        default_factory=lambda: ["alternatives", "related", "used_in"]
    )
    manual_connections: Dict[str, List[str]] = Field(default_factory=dict)

    # Anti-categories metadata (tag-based organization)
    framework_tags: List[str] = Field(default_factory=list)
    task_tags: List[str] = Field(default_factory=list)
    complexity_tags: List[str] = Field(default_factory=list)
    domain_tags: List[str] = Field(default_factory=list)
    pattern_tags: List[str] = Field(default_factory=list)
    integration_tags: List[str] = Field(default_factory=list)
    quality_tags: List[str] = Field(default_factory=list)
    data_tags: List[str] = Field(default_factory=list)

    # Manual linking metadata
    curated_connections: Dict[str, str] = Field(
        default_factory=dict
    )  # connection_id -> annotation
    connection_confidence: Dict[str, float] = Field(default_factory=dict)

    # Dual-form structure metadata
    creation_context: str = ""
    usage_frequency: str = "unknown"
    stability: str = "experimental"
    maintenance_burden: str = "unknown"

    # Discovery metadata
    estimated_runtime: str = "unknown"
    resource_requirements: str = "unknown"
    use_cases: List[str] = Field(default_factory=list)
    skill_level: str = "unknown"

    @field_validator("atomic_id")
    @classmethod
    def validate_atomic_id(cls, v):
        """Validate atomic_id is non-empty string."""
        if not v or not isinstance(v, str):
            raise ValueError("atomic_id must be a non-empty string")
        return v

    @field_validator("single_responsibility")
    @classmethod
    def validate_single_responsibility(cls, v):
        """Validate single_responsibility is concise."""
        if len(v.split()) > 15:
            logger.warning(f"Single responsibility is verbose (>15 words)")
        return v

    @model_validator(mode="before")
    @classmethod
    def set_default_pattern_tags(cls, values):
        """Set default pattern tags if none provided."""
        if isinstance(values, dict) and not values.get("pattern_tags"):
            values["pattern_tags"] = ["atomic_workflow", "independent"]
        return values

    def add_connection(
        self,
        target_id: str,
        connection_type: str,
        annotation: str,
        confidence: float = 1.0,
    ) -> None:
        """Add a manual connection following Zettelkasten principles."""
        if connection_type not in self.connection_types:
            raise ValueError(f"Invalid connection type: {connection_type}")

        # Add to manual connections
        if connection_type not in self.manual_connections:
            self.manual_connections[connection_type] = []

        if target_id not in self.manual_connections[connection_type]:
            self.manual_connections[connection_type].append(target_id)

        # Add annotation and confidence
        self.curated_connections[target_id] = annotation
        self.connection_confidence[target_id] = confidence

    def get_all_tags(self) -> Dict[str, List[str]]:
        """Get all tags organized by category."""
        return {
            "framework_tags": self.framework_tags,
            "task_tags": self.task_tags,
            "complexity_tags": self.complexity_tags,
            "domain_tags": self.domain_tags,
            "pattern_tags": self.pattern_tags,
            "integration_tags": self.integration_tags,
            "quality_tags": self.quality_tags,
            "data_tags": self.data_tags,
        }

    def get_flat_tags(self) -> List[str]:
        """Get all tags as a flat list."""
        all_tags = []
        for tag_list in self.get_all_tags().values():
            all_tags.extend(tag_list)
        return list(set(all_tags))  # Remove duplicates


class EnhancedDAGMetadata(DAGMetadata):
    """
    Enhanced DAGMetadata with Zettelkasten principles.

    Inherits from DAGMetadata and extends it with Zettelkasten-specific
    metadata while maintaining full backward compatibility.
    """

    # Zettelkasten extensions (inherits all DAGMetadata fields)
    zettelkasten_metadata: Optional[ZettelkastenMetadata] = Field(
        default=None, description="Zettelkasten-specific metadata"
    )

    # Override field validators to handle enum conversion
    @field_validator("complexity", mode="before")
    @classmethod
    def validate_complexity(cls, v):
        """Convert string complexity to enum if needed, then validate with parent."""
        if isinstance(v, str):
            try:
                return ComplexityLevel(v)
            except ValueError:
                # Fall back to parent validation for string values
                return v
        return v

    @field_validator("framework", mode="before")
    @classmethod
    def validate_framework(cls, v):
        """Convert string framework to enum if needed, then validate with parent."""
        if isinstance(v, str):
            try:
                return PipelineFramework(v)
            except ValueError:
                # Fall back to parent validation for string values
                return v
        return v

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization to create default Zettelkasten metadata if needed."""
        if self.zettelkasten_metadata is None:
            object.__setattr__(
                self,
                "zettelkasten_metadata",
                self._create_default_zettelkasten_metadata(),
            )

    def _create_default_zettelkasten_metadata(self) -> ZettelkastenMetadata:
        """Create default Zettelkasten metadata from core metadata."""
        atomic_id = self._generate_atomic_id()

        # Handle both enum and string values for framework/complexity
        framework_value = (
            self.framework.value if hasattr(self.framework, "value") else self.framework
        )
        complexity_value = (
            self.complexity.value
            if hasattr(self.complexity, "value")
            else self.complexity
        )

        return ZettelkastenMetadata(
            atomic_id=atomic_id,
            title=self._generate_title(),
            single_responsibility=self.description,
            framework=framework_value,
            complexity=complexity_value,
            features=self.features,
            node_count=self.node_count,
            edge_count=self.edge_count,
            framework_tags=[framework_value],
            task_tags=self.features,
            complexity_tags=[complexity_value],
            pattern_tags=["atomic_workflow", "independent"],
        )

    def _generate_atomic_id(self) -> str:
        """Generate atomic ID from core metadata."""
        # Handle both enum and string values
        framework_prefix = (
            self.framework.value if hasattr(self.framework, "value") else self.framework
        )
        complexity_suffix = (
            self.complexity.value
            if hasattr(self.complexity, "value")
            else self.complexity
        )

        # Extract primary feature for ID
        primary_feature = self.features[0] if self.features else "pipeline"

        return f"{framework_prefix}_{primary_feature}_{complexity_suffix}"

    def _validate(self) -> None:
        """Validate the enhanced metadata."""
        # Validate core metadata
        if not self.description:
            raise ValueError("Description cannot be empty")

        if self.node_count <= 0:
            raise ValueError("Node count must be positive")

        if self.edge_count < 0:
            raise ValueError("Edge count must be non-negative")

        if not self.features:
            raise ValueError("Features list cannot be empty")

        # Validate Zettelkasten metadata
        if not self.zettelkasten_metadata.atomic_id:
            raise ValueError("Atomic ID cannot be empty")

    def to_registry_node(self) -> Dict[str, Any]:
        """Convert to registry node format for Zettelkasten catalog."""
        zm = self.zettelkasten_metadata

        return {
            "id": zm.atomic_id,
            "title": zm.title or self._generate_title(),
            "description": self.description,
            "atomic_properties": {
                "single_responsibility": zm.single_responsibility,
                "independence_level": zm.independence_level,
                "node_count": zm.node_count,
                "edge_count": zm.edge_count,
            },
            "zettelkasten_metadata": {
                "framework": zm.framework
                or (
                    self.framework.value
                    if hasattr(self.framework, "value")
                    else self.framework
                ),
                "complexity": zm.complexity
                or (
                    self.complexity.value
                    if hasattr(self.complexity, "value")
                    else self.complexity
                ),
                "use_case": zm.use_case or self.description,
                "features": zm.features or self.features,
                "mods_compatible": zm.mods_compatible,
            },
            "multi_dimensional_tags": {
                "framework_tags": zm.framework_tags
                or [
                    (
                        self.framework.value
                        if hasattr(self.framework, "value")
                        else self.framework
                    )
                ],
                "task_tags": zm.task_tags or self.features,
                "complexity_tags": zm.complexity_tags
                or [
                    (
                        self.complexity.value
                        if hasattr(self.complexity, "value")
                        else self.complexity
                    )
                ],
            },
            "source_file": zm.source_file,
            "migration_source": zm.migration_source,
            "connections": self._build_connections(),
            "created_date": zm.created_date,
            "priority": zm.priority,
        }

    def _generate_title(self) -> str:
        """Generate human-readable title from metadata."""
        # Handle both enum and string values
        framework_value = (
            self.framework.value if hasattr(self.framework, "value") else self.framework
        )
        complexity_value = (
            self.complexity.value
            if hasattr(self.complexity, "value")
            else self.complexity
        )

        framework_name = framework_value.replace("_", " ").title()
        primary_feature = (
            self.features[0].replace("_", " ").title() if self.features else "Pipeline"
        )
        complexity_level = complexity_value.title()

        return f"{framework_name} {primary_feature} {complexity_level}"

    def _extract_dependencies(self) -> List[str]:
        """Extract dependencies from framework and features."""
        deps = []

        # Framework dependencies
        if self.framework == PipelineFramework.XGBOOST:
            deps.extend(["xgboost", "sagemaker"])
        elif self.framework == PipelineFramework.PYTORCH:
            deps.extend(["torch", "sagemaker"])
        elif self.framework == PipelineFramework.SKLEARN:
            deps.extend(["scikit-learn", "sagemaker"])
        else:
            deps.append("sagemaker")

        # Feature-specific dependencies
        if "calibration" in self.features:
            deps.append("sklearn")
        if "evaluation" in self.features:
            deps.append("pandas")

        return list(set(deps))  # Remove duplicates

    def _build_connections(self) -> Dict[str, List[Dict[str, str]]]:
        """Build connections from manual linking metadata."""
        zm = self.zettelkasten_metadata
        connections = {"alternatives": [], "related": [], "used_in": []}

        # Build from manual connections
        for conn_type, target_ids in zm.manual_connections.items():
            if conn_type in connections:
                for target_id in target_ids:
                    annotation = zm.curated_connections.get(
                        target_id, f"Connected via {conn_type}"
                    )
                    connections[conn_type].append(
                        {"id": target_id, "annotation": annotation}
                    )

        return connections

    def add_connection(
        self,
        target_id: str,
        connection_type: str,
        annotation: str,
        confidence: float = 1.0,
    ) -> None:
        """Add a manual connection following Zettelkasten principles."""
        self.zettelkasten_metadata.add_connection(
            target_id, connection_type, annotation, confidence
        )

    def update_tags(self, tag_category: str, tags: List[str]) -> None:
        """Update tags for specific category."""
        zm = self.zettelkasten_metadata

        if hasattr(zm, f"{tag_category}_tags"):
            setattr(zm, f"{tag_category}_tags", tags)
        else:
            raise ValueError(f"Invalid tag category: {tag_category}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for serialization."""
        # Get base dict from parent class
        base_dict = super().to_dict()

        # Add Zettelkasten-specific fields
        base_dict.update(
            {
                "zettelkasten_metadata": (
                    self.zettelkasten_metadata.model_dump()
                    if self.zettelkasten_metadata
                    else None
                )
            }
        )

        return base_dict

    def to_legacy_dag_metadata(self) -> DAGMetadata:
        """Convert back to legacy DAGMetadata format for compatibility."""
        # Since we inherit from DAGMetadata, we can create a new instance with the same fields
        return DAGMetadata(
            description=self.description,
            complexity=(
                self.complexity.value
                if hasattr(self.complexity, "value")
                else self.complexity
            ),
            features=self.features,
            framework=(
                self.framework.value
                if hasattr(self.framework, "value")
                else self.framework
            ),
            node_count=self.node_count,
            edge_count=self.edge_count,
            extra_metadata=self.extra_metadata,
        )


class DAGMetadataAdapter:
    """Adapter to maintain backward compatibility with existing DAGMetadata."""

    @staticmethod
    def from_legacy_dag_metadata(legacy_metadata) -> EnhancedDAGMetadata:
        """Convert legacy DAGMetadata to EnhancedDAGMetadata."""

        # Map legacy complexity to enum
        complexity_map = {
            "simple": ComplexityLevel.SIMPLE,
            "standard": ComplexityLevel.STANDARD,
            "advanced": ComplexityLevel.ADVANCED,
            "comprehensive": ComplexityLevel.COMPREHENSIVE,
        }

        # Map legacy framework to enum
        framework_map = {
            "xgboost": PipelineFramework.XGBOOST,
            "pytorch": PipelineFramework.PYTORCH,
            "sklearn": PipelineFramework.SKLEARN,
            "generic": PipelineFramework.GENERIC,
        }

        complexity = complexity_map.get(
            legacy_metadata.complexity, ComplexityLevel.STANDARD
        )
        framework = framework_map.get(
            legacy_metadata.framework, PipelineFramework.GENERIC
        )

        # Create Zettelkasten metadata from legacy data
        atomic_id = f"{framework.value}_{legacy_metadata.features[0] if legacy_metadata.features else 'pipeline'}_{complexity.value}"

        zettelkasten_metadata = ZettelkastenMetadata(
            atomic_id=atomic_id,
            single_responsibility=legacy_metadata.description,
            input_interface=legacy_metadata.extra_metadata.get(
                "input_interface", ["data"]
            ),
            output_interface=legacy_metadata.extra_metadata.get(
                "output_interface", ["model"]
            ),
            framework_tags=[framework.value],
            task_tags=legacy_metadata.features,
            complexity_tags=[complexity.value],
        )

        return EnhancedDAGMetadata(
            description=legacy_metadata.description,
            complexity=complexity,
            features=legacy_metadata.features,
            framework=framework,
            node_count=legacy_metadata.node_count,
            edge_count=legacy_metadata.edge_count,
            zettelkasten_metadata=zettelkasten_metadata,
            **legacy_metadata.extra_metadata,
        )

    @staticmethod
    def to_legacy_dag_metadata(enhanced_metadata: EnhancedDAGMetadata):
        """Convert EnhancedDAGMetadata back to legacy format if needed."""
        return enhanced_metadata.to_legacy_dag_metadata()


def validate_enhanced_dag_metadata(metadata: EnhancedDAGMetadata) -> bool:
    """
    Validate enhanced DAG metadata for consistency.

    Args:
        metadata: EnhancedDAGMetadata instance to validate

    Returns:
        bool: True if metadata is valid

    Raises:
        ValueError: If metadata is invalid
    """
    try:
        # Core validation is handled in __init__
        metadata._validate()

        # Additional Zettelkasten-specific validation
        zm = metadata.zettelkasten_metadata

        # Validate connection types
        valid_connection_types = {"alternatives", "related", "used_in"}
        for conn_type in zm.manual_connections.keys():
            if conn_type not in valid_connection_types:
                raise ValueError(f"Invalid connection type: {conn_type}")

        # Validate that all connected pipelines have annotations
        for conn_type, target_ids in zm.manual_connections.items():
            for target_id in target_ids:
                if target_id not in zm.curated_connections:
                    logger.warning(
                        f"Missing annotation for connection {metadata.zettelkasten_metadata.atomic_id} -> {target_id}"
                    )

        return True

    except Exception as e:
        logger.error(
            f"Validation failed for {metadata.zettelkasten_metadata.atomic_id}: {e}"
        )
        raise

"""
Registry Validation Utilities

Ensure registry integrity and Zettelkasten principle compliance.
Validates the registry maintains atomicity, connection integrity,
and proper dual-form structure.
"""

import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from collections import defaultdict, Counter
from pydantic import BaseModel
from enum import Enum

from .catalog_registry import CatalogRegistry

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationIssue(BaseModel):
    """Represents a validation issue."""

    severity: ValidationSeverity
    category: str
    pipeline_id: Optional[str]
    message: str
    suggested_fix: Optional[str] = None


class AtomicityViolation(ValidationIssue):
    """Violation of atomicity principle."""

    def __init__(
        self,
        pipeline_id: str,
        violation_type: str,
        description: str,
        suggested_fix: str,
        **kwargs,
    ):
        super().__init__(
            severity=ValidationSeverity.ERROR,
            category="atomicity",
            pipeline_id=pipeline_id,
            message=f"Atomicity violation ({violation_type}): {description}",
            suggested_fix=suggested_fix,
            **kwargs,
        )


class ConnectionError(ValidationIssue):
    """Connection integrity error."""

    def __init__(
        self,
        source_id: str,
        target_id: str,
        error_type: str,
        description: str,
        **kwargs,
    ):
        super().__init__(
            severity=ValidationSeverity.ERROR,
            category="connectivity",
            pipeline_id=source_id,
            message=f"Connection error ({error_type}): {source_id} -> {target_id}: {description}",
            suggested_fix=f"Fix connection between {source_id} and {target_id}",
            **kwargs,
        )


class MetadataError(ValidationIssue):
    """Metadata completeness or consistency error."""

    def __init__(
        self,
        pipeline_id: str,
        field: str,
        description: str,
        suggested_fix: str,
        **kwargs,
    ):
        super().__init__(
            severity=ValidationSeverity.WARNING,
            category="metadata",
            pipeline_id=pipeline_id,
            message=f"Metadata issue in field '{field}': {description}",
            suggested_fix=suggested_fix,
            **kwargs,
        )


class TagConsistencyError(ValidationIssue):
    """Tag usage consistency error."""

    def __init__(
        self,
        pipeline_id: str,
        tag_category: str,
        description: str,
        suggested_fix: str,
        **kwargs,
    ):
        super().__init__(
            severity=ValidationSeverity.WARNING,
            category="tags",
            pipeline_id=pipeline_id,
            message=f"Tag consistency issue in '{tag_category}': {description}",
            suggested_fix=suggested_fix,
            **kwargs,
        )


class IndependenceError(ValidationIssue):
    """Independence claim validation error."""

    def __init__(
        self, pipeline_id: str, claim: str, evidence: str, suggested_fix: str, **kwargs
    ):
        super().__init__(
            severity=ValidationSeverity.WARNING,
            category="independence",
            pipeline_id=pipeline_id,
            message=f"Independence claim '{claim}' contradicted by: {evidence}",
            suggested_fix=suggested_fix,
            **kwargs,
        )


class ValidationReport(BaseModel):
    """Comprehensive validation report."""

    is_valid: bool
    total_issues: int
    issues_by_severity: Dict[ValidationSeverity, int]
    issues_by_category: Dict[str, int]
    all_issues: List[ValidationIssue]

    def summary(self) -> str:
        """Generate human-readable validation summary."""
        if self.is_valid:
            return "Registry validation passed with no critical issues."

        error_count = self.issues_by_severity.get(ValidationSeverity.ERROR, 0)
        warning_count = self.issues_by_severity.get(ValidationSeverity.WARNING, 0)
        info_count = self.issues_by_severity.get(ValidationSeverity.INFO, 0)

        summary_parts = []
        if error_count > 0:
            summary_parts.append(f"{error_count} errors")
        if warning_count > 0:
            summary_parts.append(f"{warning_count} warnings")
        if info_count > 0:
            summary_parts.append(f"{info_count} info items")

        return f"Registry validation found {', '.join(summary_parts)}."


class RegistryValidator:
    """
    Validate registry structure and Zettelkasten principle compliance.

    Ensures the registry maintains atomicity, connection integrity,
    and proper dual-form structure.
    """

    def __init__(self, registry: CatalogRegistry):
        """
        Initialize with registry instance.

        Args:
            registry: CatalogRegistry instance to validate
        """
        self.registry = registry
        self._validation_cache = {}
        self._cache_valid = False

    def validate_atomicity(self) -> List[AtomicityViolation]:
        """
        Validate that each pipeline represents one atomic concept.

        Returns:
            List of atomicity violations
        """
        violations = []

        try:
            all_pipelines = self.registry.get_all_pipelines()

            for pipeline_id in all_pipelines:
                node = self.registry.get_pipeline_node(pipeline_id)
                if not node:
                    continue

                # Check single responsibility
                atomic_props = node.get("atomic_properties", {})
                responsibility = atomic_props.get("single_responsibility", "")

                if not responsibility:
                    violations.append(
                        AtomicityViolation(
                            pipeline_id=pipeline_id,
                            violation_type="missing_responsibility",
                            description="No single responsibility defined",
                            suggested_fix="Define a clear, concise single responsibility",
                        )
                    )
                elif len(responsibility.split()) > 15:
                    violations.append(
                        AtomicityViolation(
                            pipeline_id=pipeline_id,
                            violation_type="verbose_responsibility",
                            description=f"Single responsibility too verbose ({len(responsibility.split())} words)",
                            suggested_fix="Simplify single responsibility to ≤15 words",
                        )
                    )

                # Check interface clarity
                input_interface = atomic_props.get("input_interface", [])
                output_interface = atomic_props.get("output_interface", [])

                if not input_interface:
                    violations.append(
                        AtomicityViolation(
                            pipeline_id=pipeline_id,
                            violation_type="missing_input_interface",
                            description="Input interface not specified",
                            suggested_fix="Define clear input interface",
                        )
                    )

                if not output_interface:
                    violations.append(
                        AtomicityViolation(
                            pipeline_id=pipeline_id,
                            violation_type="missing_output_interface",
                            description="Output interface not specified",
                            suggested_fix="Define clear output interface",
                        )
                    )

                # Check independence claims
                independence = atomic_props.get("independence", "unknown")
                side_effects = atomic_props.get("side_effects", "unknown")

                if independence == "fully_self_contained" and side_effects not in [
                    "none",
                    "creates_artifacts",
                ]:
                    violations.append(
                        AtomicityViolation(
                            pipeline_id=pipeline_id,
                            violation_type="independence_contradiction",
                            description=f"Claims full independence but has side effects: {side_effects}",
                            suggested_fix="Either reduce side effects or adjust independence claim",
                        )
                    )

        except Exception as e:
            logger.error(f"Failed to validate atomicity: {e}")

        return violations

    def validate_connections(self) -> List[ConnectionError]:
        """
        Validate connection integrity and bidirectionality.

        Returns:
            List of connection errors
        """
        errors = []

        try:
            all_pipelines = set(self.registry.get_all_pipelines())

            for pipeline_id in all_pipelines:
                connections = self.registry.get_pipeline_connections(pipeline_id)

                for conn_type, conn_list in connections.items():
                    # Validate connection type
                    valid_types = ["alternatives", "related", "used_in"]
                    if conn_type not in valid_types:
                        errors.append(
                            ConnectionError(
                                source_id=pipeline_id,
                                target_id="N/A",
                                error_type="invalid_type",
                                description=f"Invalid connection type: {conn_type}",
                            )
                        )
                        continue

                    for conn in conn_list:
                        target_id = conn.get("id", "")
                        annotation = conn.get("annotation", "")

                        # Check target exists
                        if target_id not in all_pipelines:
                            errors.append(
                                ConnectionError(
                                    source_id=pipeline_id,
                                    target_id=target_id,
                                    error_type="missing_target",
                                    description="Target pipeline does not exist",
                                )
                            )

                        # Check annotation exists
                        if not annotation:
                            errors.append(
                                ConnectionError(
                                    source_id=pipeline_id,
                                    target_id=target_id,
                                    error_type="missing_annotation",
                                    description="Connection lacks annotation",
                                )
                            )

                        # Check for self-references
                        if target_id == pipeline_id:
                            errors.append(
                                ConnectionError(
                                    source_id=pipeline_id,
                                    target_id=target_id,
                                    error_type="self_reference",
                                    description="Pipeline connects to itself",
                                )
                            )

            # Check for orphaned connections in tag index
            registry_data = self.registry.load_registry()
            tag_index = registry_data.get("tag_index", {})

            for category, tag_dict in tag_index.items():
                for tag, pipeline_list in tag_dict.items():
                    for pipeline_id in pipeline_list:
                        if pipeline_id not in all_pipelines:
                            errors.append(
                                ConnectionError(
                                    source_id="tag_index",
                                    target_id=pipeline_id,
                                    error_type="orphaned_tag_reference",
                                    description=f"Tag index references non-existent pipeline in {category}:{tag}",
                                )
                            )

        except Exception as e:
            logger.error(f"Failed to validate connections: {e}")

        return errors

    def validate_metadata_completeness(self) -> List[MetadataError]:
        """
        Validate that all required metadata is present.

        Returns:
            List of metadata errors
        """
        errors = []

        try:
            all_pipelines = self.registry.get_all_pipelines()

            for pipeline_id in all_pipelines:
                node = self.registry.get_pipeline_node(pipeline_id)
                if not node:
                    errors.append(
                        MetadataError(
                            pipeline_id=pipeline_id,
                            field="node",
                            description="Pipeline node not found",
                            suggested_fix="Ensure pipeline is properly registered",
                        )
                    )
                    continue

                # Check required top-level fields
                required_fields = [
                    "id",
                    "title",
                    "description",
                    "atomic_properties",
                    "zettelkasten_metadata",
                ]
                for field in required_fields:
                    if field not in node:
                        errors.append(
                            MetadataError(
                                pipeline_id=pipeline_id,
                                field=field,
                                description=f"Missing required field: {field}",
                                suggested_fix=f"Add {field} to pipeline metadata",
                            )
                        )

                # Check atomic properties completeness
                atomic_props = node.get("atomic_properties", {})
                required_atomic_fields = [
                    "single_responsibility",
                    "input_interface",
                    "output_interface",
                ]
                for field in required_atomic_fields:
                    if field not in atomic_props:
                        errors.append(
                            MetadataError(
                                pipeline_id=pipeline_id,
                                field=f"atomic_properties.{field}",
                                description=f"Missing atomic property: {field}",
                                suggested_fix=f"Add {field} to atomic properties",
                            )
                        )

                # Check zettelkasten metadata completeness
                zk_metadata = node.get("zettelkasten_metadata", {})
                required_zk_fields = ["framework", "complexity"]
                for field in required_zk_fields:
                    if field not in zk_metadata:
                        errors.append(
                            MetadataError(
                                pipeline_id=pipeline_id,
                                field=f"zettelkasten_metadata.{field}",
                                description=f"Missing zettelkasten metadata: {field}",
                                suggested_fix=f"Add {field} to zettelkasten metadata",
                            )
                        )

                # Check multi-dimensional tags
                tags = node.get("multi_dimensional_tags", {})
                required_tag_categories = [
                    "framework_tags",
                    "task_tags",
                    "complexity_tags",
                ]
                for category in required_tag_categories:
                    if category not in tags or not tags[category]:
                        errors.append(
                            MetadataError(
                                pipeline_id=pipeline_id,
                                field=f"multi_dimensional_tags.{category}",
                                description=f"Missing or empty tag category: {category}",
                                suggested_fix=f"Add tags to {category}",
                            )
                        )

                # Check discovery metadata
                discovery_meta = node.get("discovery_metadata", {})
                important_discovery_fields = [
                    "estimated_runtime",
                    "resource_requirements",
                    "skill_level",
                ]
                for field in important_discovery_fields:
                    if (
                        field not in discovery_meta
                        or discovery_meta[field] == "unknown"
                    ):
                        errors.append(
                            MetadataError(
                                pipeline_id=pipeline_id,
                                field=f"discovery_metadata.{field}",
                                description=f"Discovery metadata field not specified: {field}",
                                suggested_fix=f"Specify {field} in discovery metadata",
                            )
                        )

        except Exception as e:
            logger.error(f"Failed to validate metadata completeness: {e}")

        return errors

    def validate_tag_consistency(self) -> List[TagConsistencyError]:
        """
        Validate tag usage consistency across pipelines.

        Returns:
            List of tag consistency errors
        """
        errors = []

        try:
            all_pipelines = self.registry.get_all_pipelines()

            # Collect all tags and their usage
            tag_usage = defaultdict(
                lambda: defaultdict(set)
            )  # category -> tag -> set of pipelines

            for pipeline_id in all_pipelines:
                node = self.registry.get_pipeline_node(pipeline_id)
                if not node:
                    continue

                tags = node.get("multi_dimensional_tags", {})
                zk_metadata = node.get("zettelkasten_metadata", {})

                # Check framework tag consistency
                framework = zk_metadata.get("framework", "")
                framework_tags = tags.get("framework_tags", [])

                if framework and framework not in framework_tags:
                    errors.append(
                        TagConsistencyError(
                            pipeline_id=pipeline_id,
                            tag_category="framework_tags",
                            description=f"Framework '{framework}' not in framework_tags: {framework_tags}",
                            suggested_fix=f"Add '{framework}' to framework_tags",
                        )
                    )

                # Check complexity tag consistency
                complexity = zk_metadata.get("complexity", "")
                complexity_tags = tags.get("complexity_tags", [])

                if complexity and complexity not in complexity_tags:
                    errors.append(
                        TagConsistencyError(
                            pipeline_id=pipeline_id,
                            tag_category="complexity_tags",
                            description=f"Complexity '{complexity}' not in complexity_tags: {complexity_tags}",
                            suggested_fix=f"Add '{complexity}' to complexity_tags",
                        )
                    )

                # Collect tag usage for global consistency checks
                for category, tag_list in tags.items():
                    for tag in tag_list:
                        tag_usage[category][tag].add(pipeline_id)

            # Check for inconsistent tag usage patterns
            for category, category_tags in tag_usage.items():
                for tag, pipelines in category_tags.items():
                    if len(pipelines) == 1:
                        # Single-use tags might indicate typos or inconsistency
                        pipeline_id = list(pipelines)[0]

                        # Look for similar tags that might be typos
                        similar_tags = []
                        for other_tag in category_tags.keys():
                            if other_tag != tag and self._are_similar_tags(
                                tag, other_tag
                            ):
                                similar_tags.append(other_tag)

                        if similar_tags:
                            errors.append(
                                TagConsistencyError(
                                    pipeline_id=pipeline_id,
                                    tag_category=category,
                                    description=f"Tag '{tag}' used only once, similar to: {similar_tags}",
                                    suggested_fix=f"Consider using consistent tag: {similar_tags[0]}",
                                )
                            )

        except Exception as e:
            logger.error(f"Failed to validate tag consistency: {e}")

        return errors

    def validate_independence_claims(self) -> List[IndependenceError]:
        """
        Validate that pipelines marked as independent truly are.

        Returns:
            List of independence errors
        """
        errors = []

        try:
            all_pipelines = self.registry.get_all_pipelines()

            for pipeline_id in all_pipelines:
                node = self.registry.get_pipeline_node(pipeline_id)
                if not node:
                    continue

                atomic_props = node.get("atomic_properties", {})
                independence = atomic_props.get("independence", "unknown")

                if independence == "fully_self_contained":
                    # Check for evidence that contradicts independence

                    # Check side effects
                    side_effects = atomic_props.get("side_effects", "unknown")
                    if side_effects not in ["none", "creates_artifacts"]:
                        errors.append(
                            IndependenceError(
                                pipeline_id=pipeline_id,
                                claim="fully_self_contained",
                                evidence=f"has side effects: {side_effects}",
                                suggested_fix="Reduce side effects or adjust independence claim",
                            )
                        )

                    # Check dependencies
                    dependencies = atomic_props.get("dependencies", [])
                    if len(dependencies) > 3:  # More than basic framework dependencies
                        errors.append(
                            IndependenceError(
                                pipeline_id=pipeline_id,
                                claim="fully_self_contained",
                                evidence=f"has many dependencies: {dependencies}",
                                suggested_fix="Reduce dependencies or adjust independence claim",
                            )
                        )

                    # Check for complex input requirements
                    input_interface = atomic_props.get("input_interface", [])
                    if len(input_interface) > 5:  # Many input requirements
                        errors.append(
                            IndependenceError(
                                pipeline_id=pipeline_id,
                                claim="fully_self_contained",
                                evidence=f"requires many inputs: {input_interface}",
                                suggested_fix="Simplify input interface or adjust independence claim",
                            )
                        )

        except Exception as e:
            logger.error(f"Failed to validate independence claims: {e}")

        return errors

    def generate_validation_report(self) -> ValidationReport:
        """
        Generate comprehensive validation report.

        Returns:
            Complete validation report
        """
        try:
            all_issues = []

            # Run all validations
            atomicity_violations = self.validate_atomicity()
            connection_errors = self.validate_connections()
            metadata_errors = self.validate_metadata_completeness()
            tag_consistency_errors = self.validate_tag_consistency()
            independence_errors = self.validate_independence_claims()

            # Combine all issues
            all_issues.extend(atomicity_violations)
            all_issues.extend(connection_errors)
            all_issues.extend(metadata_errors)
            all_issues.extend(tag_consistency_errors)
            all_issues.extend(independence_errors)

            # Count issues by severity and category
            issues_by_severity = Counter()
            issues_by_category = Counter()

            for issue in all_issues:
                issues_by_severity[issue.severity] += 1
                issues_by_category[issue.category] += 1

            # Determine if registry is valid (no errors, warnings are acceptable)
            error_count = issues_by_severity.get(ValidationSeverity.ERROR, 0)
            is_valid = error_count == 0

            return ValidationReport(
                is_valid=is_valid,
                total_issues=len(all_issues),
                issues_by_severity=dict(issues_by_severity),
                issues_by_category=dict(issues_by_category),
                all_issues=all_issues,
            )

        except Exception as e:
            logger.error(f"Failed to generate validation report: {e}")
            return ValidationReport(
                is_valid=False,
                total_issues=1,
                issues_by_severity={ValidationSeverity.ERROR: 1},
                issues_by_category={"system": 1},
                all_issues=[
                    ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        category="system",
                        pipeline_id=None,
                        message=f"Validation system error: {e}",
                        suggested_fix="Check system logs and fix validation system",
                    )
                ],
            )

    def validate_zettelkasten_principles(self) -> Dict[str, Any]:
        """
        Validate adherence to core Zettelkasten principles.

        Returns:
            Dictionary with principle compliance analysis
        """
        try:
            all_pipelines = self.registry.get_all_pipelines()
            total_pipelines = len(all_pipelines)

            if total_pipelines == 0:
                return {"error": "No pipelines found in registry"}

            # Principle 1: Atomicity
            atomic_pipelines = 0
            for pipeline_id in all_pipelines:
                node = self.registry.get_pipeline_node(pipeline_id)
                if node:
                    atomic_props = node.get("atomic_properties", {})
                    if (
                        atomic_props.get("single_responsibility")
                        and atomic_props.get("input_interface")
                        and atomic_props.get("output_interface")
                    ):
                        atomic_pipelines += 1

            atomicity_score = atomic_pipelines / total_pipelines

            # Principle 2: Connectivity
            connected_pipelines = 0
            total_connections = 0
            for pipeline_id in all_pipelines:
                connections = self.registry.get_pipeline_connections(pipeline_id)
                connection_count = sum(
                    len(conn_list) for conn_list in connections.values()
                )
                if connection_count > 0:
                    connected_pipelines += 1
                total_connections += connection_count

            connectivity_score = connected_pipelines / total_pipelines
            avg_connections = total_connections / total_pipelines

            # Principle 3: Anti-categories (tag diversity)
            registry_data = self.registry.load_registry()
            tag_index = registry_data.get("tag_index", {})
            tag_categories = len(tag_index)
            total_unique_tags = sum(
                len(category_tags) for category_tags in tag_index.values()
            )

            tag_diversity_score = min(
                tag_categories / 8, 1.0
            )  # Expect ~8 tag categories

            # Principle 4: Manual linking (curated connections)
            curated_connections = 0
            for pipeline_id in all_pipelines:
                connections = self.registry.get_pipeline_connections(pipeline_id)
                for conn_list in connections.values():
                    for conn in conn_list:
                        if conn.get("annotation"):  # Has human annotation
                            curated_connections += 1

            curation_score = min(curated_connections / max(total_connections, 1), 1.0)

            # Principle 5: Dual-form structure (metadata separation)
            well_structured_pipelines = 0
            for pipeline_id in all_pipelines:
                node = self.registry.get_pipeline_node(pipeline_id)
                if node:
                    # Check for proper separation of concerns
                    has_atomic_props = bool(node.get("atomic_properties"))
                    has_zk_metadata = bool(node.get("zettelkasten_metadata"))
                    has_tags = bool(node.get("multi_dimensional_tags"))
                    has_discovery = bool(node.get("discovery_metadata"))

                    if (
                        has_atomic_props
                        and has_zk_metadata
                        and has_tags
                        and has_discovery
                    ):
                        well_structured_pipelines += 1

            structure_score = well_structured_pipelines / total_pipelines

            # Overall compliance score
            overall_score = (
                atomicity_score * 0.25
                + connectivity_score * 0.2
                + tag_diversity_score * 0.2
                + curation_score * 0.15
                + structure_score * 0.2
            )

            return {
                "overall_compliance": overall_score,
                "principle_scores": {
                    "atomicity": atomicity_score,
                    "connectivity": connectivity_score,
                    "anti_categories": tag_diversity_score,
                    "manual_linking": curation_score,
                    "dual_form_structure": structure_score,
                },
                "metrics": {
                    "total_pipelines": total_pipelines,
                    "atomic_pipelines": atomic_pipelines,
                    "connected_pipelines": connected_pipelines,
                    "average_connections": avg_connections,
                    "tag_categories": tag_categories,
                    "unique_tags": total_unique_tags,
                    "curated_connections": curated_connections,
                    "well_structured_pipelines": well_structured_pipelines,
                },
                "recommendations": self._generate_principle_recommendations(
                    atomicity_score,
                    connectivity_score,
                    tag_diversity_score,
                    curation_score,
                    structure_score,
                ),
            }

        except Exception as e:
            logger.error(f"Failed to validate Zettelkasten principles: {e}")
            return {"error": str(e)}

    def _are_similar_tags(self, tag1: str, tag2: str) -> bool:
        """Check if two tags are similar (potential typos)."""
        # Simple similarity check based on edit distance
        if abs(len(tag1) - len(tag2)) > 2:
            return False

        # Check for common prefixes/suffixes
        if len(tag1) >= 4 and len(tag2) >= 4:
            if tag1[:3] == tag2[:3] or tag1[-3:] == tag2[-3:]:
                return True

        # Check for single character differences
        if len(tag1) == len(tag2):
            diff_count = sum(c1 != c2 for c1, c2 in zip(tag1, tag2))
            return diff_count <= 1

        return False

    def _generate_principle_recommendations(
        self,
        atomicity: float,
        connectivity: float,
        anti_categories: float,
        manual_linking: float,
        dual_form: float,
    ) -> List[str]:
        """Generate recommendations for improving principle compliance."""
        recommendations = []

        if atomicity < 0.8:
            recommendations.append(
                "Improve atomicity by defining clear single responsibilities and interfaces for all pipelines"
            )

        if connectivity < 0.6:
            recommendations.append(
                "Increase connectivity by adding more connections between related pipelines"
            )

        if anti_categories < 0.7:
            recommendations.append(
                "Enhance tag diversity by using more tag categories and specific tags"
            )

        if manual_linking < 0.8:
            recommendations.append(
                "Improve manual linking by adding meaningful annotations to all connections"
            )

        if dual_form < 0.9:
            recommendations.append(
                "Complete dual-form structure by ensuring all pipelines have comprehensive metadata"
            )

        return recommendations

    def clear_cache(self) -> None:
        """Clear the internal validation cache."""
        self._validation_cache = {}
        self._cache_valid = False
        logger.debug("Registry validator cache cleared")

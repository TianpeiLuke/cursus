"""
Phase 2: Logical Name Matching System for Pipeline Runtime Testing

This module implements intelligent path matching between script outputs and inputs
using semantic similarity, alias systems, and topological execution ordering.
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from pydantic import BaseModel, Field
from enum import Enum

# Import existing semantic matching infrastructure
from ...core.deps.semantic_matcher import SemanticMatcher
from .runtime_models import ScriptExecutionSpec

logger = logging.getLogger(__name__)


class MatchType(str, Enum):
    """Types of path matches between source outputs and destination inputs"""

    EXACT_LOGICAL = "exact_logical"
    LOGICAL_TO_ALIAS = "logical_to_alias"
    ALIAS_TO_LOGICAL = "alias_to_logical"
    ALIAS_TO_ALIAS = "alias_to_alias"
    SEMANTIC = "semantic"


class PathSpec(BaseModel):
    """Enhanced path specification with alias support following OutputSpec pattern"""

    logical_name: str = Field(..., description="Primary logical name")
    path: str = Field(..., description="File system path")
    aliases: List[str] = Field(default_factory=list, description="Alternative names")

    def matches_name_or_alias(self, name: str) -> bool:
        """Check if name matches logical_name or any alias"""
        return name == self.logical_name or name in self.aliases


class PathMatch(BaseModel):
    """Represents a successful match between source output and destination input"""

    source_logical_name: str = Field(..., description="Source output logical name")
    dest_logical_name: str = Field(..., description="Destination input logical name")
    match_type: MatchType = Field(..., description="Type of match found")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)")
    semantic_details: Optional[Dict[str, Any]] = Field(
        None, description="Detailed semantic matching information"
    )
    matched_source_name: str = Field(..., description="Actual source name that matched")
    matched_dest_name: str = Field(
        ..., description="Actual destination name that matched"
    )


class EnhancedScriptExecutionSpec(ScriptExecutionSpec):
    """Enhanced ScriptExecutionSpec with alias system support"""

    # Enhanced path specifications with alias support
    input_path_specs: Dict[str, PathSpec] = Field(
        default_factory=dict, description="Input path specifications with aliases"
    )
    output_path_specs: Dict[str, PathSpec] = Field(
        default_factory=dict, description="Output path specifications with aliases"
    )

    def model_post_init(self, __context) -> None:
        """Post-initialization to sync path specs with inherited path fields"""
        super().model_post_init(__context)

        # If input_path_specs is provided but input_paths is empty, populate input_paths
        if self.input_path_specs and not self.input_paths:
            self.input_paths = {
                logical_name: spec.path
                for logical_name, spec in self.input_path_specs.items()
            }

        # If output_path_specs is provided but output_paths is empty, populate output_paths
        if self.output_path_specs and not self.output_paths:
            self.output_paths = {
                logical_name: spec.path
                for logical_name, spec in self.output_path_specs.items()
            }

        # If input_paths is provided but input_path_specs is empty, populate input_path_specs
        if self.input_paths and not self.input_path_specs:
            self.input_path_specs = {
                logical_name: PathSpec(logical_name=logical_name, path=path, aliases=[])
                for logical_name, path in self.input_paths.items()
            }

        # If output_paths is provided but output_path_specs is empty, populate output_path_specs
        if self.output_paths and not self.output_path_specs:
            self.output_path_specs = {
                logical_name: PathSpec(logical_name=logical_name, path=path, aliases=[])
                for logical_name, path in self.output_paths.items()
            }

    @classmethod
    def from_script_execution_spec(
        cls,
        original_spec,
        input_aliases: Optional[Dict[str, List[str]]] = None,
        output_aliases: Optional[Dict[str, List[str]]] = None,
    ) -> "EnhancedScriptExecutionSpec":
        """Create enhanced spec from original ScriptExecutionSpec"""
        input_aliases = input_aliases or {}
        output_aliases = output_aliases or {}

        # Convert input_paths to input_path_specs
        input_path_specs = {}
        for logical_name, path in original_spec.input_paths.items():
            input_path_specs[logical_name] = PathSpec(
                logical_name=logical_name,
                path=path,
                aliases=input_aliases.get(logical_name, []),
            )

        # Convert output_paths to output_path_specs
        output_path_specs = {}
        for logical_name, path in original_spec.output_paths.items():
            output_path_specs[logical_name] = PathSpec(
                logical_name=logical_name,
                path=path,
                aliases=output_aliases.get(logical_name, []),
            )

        return cls(
            script_name=original_spec.script_name,
            step_name=original_spec.step_name,
            script_path=original_spec.script_path,
            input_path_specs=input_path_specs,
            output_path_specs=output_path_specs,
            environ_vars=original_spec.environ_vars,
            job_args=original_spec.job_args,
            last_updated=getattr(original_spec, "last_updated", None),
            user_notes=getattr(original_spec, "user_notes", None),
        )


class PathMatcher:
    """Handles logical name matching between ScriptExecutionSpecs using semantic matching"""

    def __init__(self, semantic_threshold: float = 0.7):
        """
        Initialize PathMatcher with semantic matching capabilities

        Args:
            semantic_threshold: Minimum similarity score for semantic matches
        """
        self.semantic_matcher = SemanticMatcher()
        self.semantic_threshold = semantic_threshold

    def find_path_matches(
        self,
        source_spec: EnhancedScriptExecutionSpec,
        dest_spec: EnhancedScriptExecutionSpec,
    ) -> List[PathMatch]:
        """
        Find matches between source outputs and destination inputs

        Matching Priority:
        1. Exact logical name match
        2. Logical name to alias match
        3. Alias to logical name match
        4. Alias to alias match
        5. Semantic similarity match (above threshold)

        Args:
            source_spec: Source script specification with outputs
            dest_spec: Destination script specification with inputs

        Returns:
            List of PathMatch objects sorted by confidence (highest first)
        """
        matches = []

        for src_logical_name, src_path_spec in source_spec.output_path_specs.items():
            for dest_logical_name, dest_path_spec in dest_spec.input_path_specs.items():

                # Level 1: Exact logical name match
                if src_path_spec.logical_name == dest_path_spec.logical_name:
                    matches.append(
                        PathMatch(
                            source_logical_name=src_logical_name,
                            dest_logical_name=dest_logical_name,
                            match_type=MatchType.EXACT_LOGICAL,
                            confidence=1.0,
                            matched_source_name=src_path_spec.logical_name,
                            matched_dest_name=dest_path_spec.logical_name,
                        )
                    )
                    continue

                # Level 2-4: Check all name/alias combinations
                best_alias_match = self._find_best_alias_match(
                    src_path_spec, dest_path_spec, src_logical_name, dest_logical_name
                )
                if best_alias_match:
                    matches.append(best_alias_match)
                    continue

                # Level 5: Semantic similarity using existing SemanticMatcher
                similarity = self.semantic_matcher.calculate_similarity(
                    src_path_spec.logical_name, dest_path_spec.logical_name
                )
                if similarity >= self.semantic_threshold:
                    semantic_details = self.semantic_matcher.explain_similarity(
                        src_path_spec.logical_name, dest_path_spec.logical_name
                    )
                    matches.append(
                        PathMatch(
                            source_logical_name=src_logical_name,
                            dest_logical_name=dest_logical_name,
                            match_type=MatchType.SEMANTIC,
                            confidence=similarity,
                            semantic_details=semantic_details,
                            matched_source_name=src_path_spec.logical_name,
                            matched_dest_name=dest_path_spec.logical_name,
                        )
                    )

        # Sort by confidence (highest first)
        matches.sort(key=lambda m: m.confidence, reverse=True)

        logger.debug(
            f"Found {len(matches)} path matches between {source_spec.script_name} and {dest_spec.script_name}"
        )
        for match in matches[:3]:  # Log top 3 matches
            logger.debug(
                f"  {match.match_type.value}: {match.matched_source_name} -> {match.matched_dest_name} "
                f"(confidence: {match.confidence:.3f})"
            )

        return matches

    def _find_best_alias_match(
        self,
        src_path_spec: PathSpec,
        dest_path_spec: PathSpec,
        src_logical_name: str,
        dest_logical_name: str,
    ) -> Optional[PathMatch]:
        """
        Find the best alias match between source and destination path specs

        Args:
            src_path_spec: Source path specification
            dest_path_spec: Destination path specification
            src_logical_name: Source logical name for the match result
            dest_logical_name: Destination logical name for the match result

        Returns:
            PathMatch if a good alias match is found, None otherwise
        """
        # Level 2: Logical name to alias match
        for dest_alias in dest_path_spec.aliases:
            if src_path_spec.logical_name == dest_alias:
                return PathMatch(
                    source_logical_name=src_logical_name,
                    dest_logical_name=dest_logical_name,
                    match_type=MatchType.LOGICAL_TO_ALIAS,
                    confidence=0.95,
                    matched_source_name=src_path_spec.logical_name,
                    matched_dest_name=dest_alias,
                )

        # Level 3: Alias to logical name match
        for src_alias in src_path_spec.aliases:
            if src_alias == dest_path_spec.logical_name:
                return PathMatch(
                    source_logical_name=src_logical_name,
                    dest_logical_name=dest_logical_name,
                    match_type=MatchType.ALIAS_TO_LOGICAL,
                    confidence=0.95,
                    matched_source_name=src_alias,
                    matched_dest_name=dest_path_spec.logical_name,
                )

        # Level 4: Alias to alias match
        for src_alias in src_path_spec.aliases:
            for dest_alias in dest_path_spec.aliases:
                if src_alias == dest_alias:
                    return PathMatch(
                        source_logical_name=src_logical_name,
                        dest_logical_name=dest_logical_name,
                        match_type=MatchType.ALIAS_TO_ALIAS,
                        confidence=0.9,
                        matched_source_name=src_alias,
                        matched_dest_name=dest_alias,
                    )

        return None

    def generate_matching_report(self, matches: List[PathMatch]) -> Dict[str, Any]:
        """
        Generate a detailed report of path matching results

        Args:
            matches: List of PathMatch objects

        Returns:
            Dictionary containing detailed matching information
        """
        if not matches:
            return {
                "total_matches": 0,
                "match_summary": "No matches found",
                "recommendations": [
                    "Check logical names and aliases for compatibility"
                ],
            }

        # Group matches by type
        matches_by_type = {}
        for match in matches:
            match_type = match.match_type.value
            if match_type not in matches_by_type:
                matches_by_type[match_type] = []
            matches_by_type[match_type].append(match)

        # Calculate statistics
        avg_confidence = sum(m.confidence for m in matches) / len(matches)
        high_confidence_matches = [m for m in matches if m.confidence >= 0.8]

        report = {
            "total_matches": len(matches),
            "high_confidence_matches": len(high_confidence_matches),
            "average_confidence": round(avg_confidence, 3),
            "matches_by_type": {
                match_type: len(match_list)
                for match_type, match_list in matches_by_type.items()
            },
            "best_matches": [
                {
                    "source": match.matched_source_name,
                    "destination": match.matched_dest_name,
                    "type": match.match_type.value,
                    "confidence": round(match.confidence, 3),
                }
                for match in matches[:5]  # Top 5 matches
            ],
        }

        # Add recommendations
        recommendations = []
        if avg_confidence < 0.6:
            recommendations.append("Consider adding aliases to improve matching")
        if len(high_confidence_matches) == 0:
            recommendations.append(
                "No high-confidence matches found - review logical names"
            )
        if MatchType.SEMANTIC.value in matches_by_type:
            recommendations.append(
                "Semantic matches found - consider standardizing naming conventions"
            )

        report["recommendations"] = recommendations

        return report


class EnhancedDataCompatibilityResult(BaseModel):
    """Enhanced result model with path matching information"""

    script_a: str
    script_b: str
    compatible: bool
    compatibility_issues: List[str] = Field(default_factory=list)
    data_format_a: Optional[str] = None
    data_format_b: Optional[str] = None

    # Enhanced matching information
    path_matches: List[PathMatch] = Field(default_factory=list)
    matching_details: Optional[Dict[str, Any]] = None
    files_tested: List[str] = Field(default_factory=list)
    execution_time: float = Field(
        default=0.0, description="Total execution time for compatibility test"
    )


class TopologicalExecutor:
    """Handles topological execution ordering for pipeline testing"""

    def __init__(self):
        """Initialize the topological executor"""
        pass

    def get_execution_order(self, dag) -> List[str]:
        """
        Get topological execution order from DAG

        Args:
            dag: PipelineDAG object

        Returns:
            List of node names in topological order

        Raises:
            ValueError: If DAG contains cycles
        """
        try:
            # Use existing topological_sort method from PipelineDAG
            return dag.topological_sort()
        except Exception as e:
            logger.error(f"Failed to get topological order: {str(e)}")
            raise ValueError(f"DAG topology error: {str(e)}")

    def validate_dag_structure(self, dag, script_specs: Dict[str, Any]) -> List[str]:
        """
        Validate DAG structure and script spec alignment

        Args:
            dag: PipelineDAG object
            script_specs: Dictionary of script specifications

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check that all DAG nodes have corresponding script specs
        for node in dag.nodes:
            if node not in script_specs:
                errors.append(f"No ScriptExecutionSpec found for DAG node: {node}")

        # Check that all script specs correspond to DAG nodes
        for spec_name in script_specs.keys():
            if spec_name not in dag.nodes:
                errors.append(
                    f"ScriptExecutionSpec '{spec_name}' not found in DAG nodes"
                )

        # Check for isolated nodes (nodes with no edges)
        nodes_with_edges = set()
        for src, dst in dag.edges:
            nodes_with_edges.add(src)
            nodes_with_edges.add(dst)

        isolated_nodes = set(dag.nodes) - nodes_with_edges
        if isolated_nodes and len(dag.nodes) > 1:
            errors.append(f"Isolated nodes found (no connections): {isolated_nodes}")

        return errors


class LogicalNameMatchingTester:
    """Enhanced runtime tester with logical name matching capabilities"""

    def __init__(self, semantic_threshold: float = 0.7):
        """
        Initialize the enhanced tester

        Args:
            semantic_threshold: Minimum similarity score for semantic matches
        """
        self.path_matcher = PathMatcher(semantic_threshold)
        self.topological_executor = TopologicalExecutor()

    def test_data_compatibility_with_logical_matching(
        self,
        spec_a: EnhancedScriptExecutionSpec,
        spec_b: EnhancedScriptExecutionSpec,
        output_files: List[Path],
    ) -> EnhancedDataCompatibilityResult:
        """
        Test data compatibility between scripts using intelligent path matching

        Args:
            spec_a: Source script specification
            spec_b: Destination script specification
            output_files: List of actual output files from script A

        Returns:
            EnhancedDataCompatibilityResult with detailed matching information
        """
        import time

        start_time = time.time()

        try:
            # Find logical name matches
            path_matches = self.path_matcher.find_path_matches(spec_a, spec_b)

            if not path_matches:
                return EnhancedDataCompatibilityResult(
                    script_a=spec_a.script_name,
                    script_b=spec_b.script_name,
                    compatible=False,
                    compatibility_issues=[
                        "No matching logical names found between source outputs and destination inputs"
                    ],
                    path_matches=[],
                    matching_details=self.path_matcher.generate_matching_report([]),
                    execution_time=time.time() - start_time,
                )

            # Create file mapping based on matches and available files
            file_mapping = self._create_file_mapping(path_matches, output_files)

            if not file_mapping:
                return EnhancedDataCompatibilityResult(
                    script_a=spec_a.script_name,
                    script_b=spec_b.script_name,
                    compatible=False,
                    compatibility_issues=[
                        "No output files could be mapped to destination inputs"
                    ],
                    path_matches=path_matches,
                    matching_details=self.path_matcher.generate_matching_report(
                        path_matches
                    ),
                    files_tested=[f.name for f in output_files],
                    execution_time=time.time() - start_time,
                )

            # Generate detailed matching report
            matching_details = self.path_matcher.generate_matching_report(path_matches)

            return EnhancedDataCompatibilityResult(
                script_a=spec_a.script_name,
                script_b=spec_b.script_name,
                compatible=True,
                compatibility_issues=[],
                path_matches=path_matches,
                matching_details=matching_details,
                files_tested=[f.name for f in output_files],
                data_format_a=self._detect_primary_format(output_files),
                data_format_b=self._detect_primary_format(
                    output_files
                ),  # Same format for successful transfer
                execution_time=time.time() - start_time,
            )

        except Exception as e:
            return EnhancedDataCompatibilityResult(
                script_a=spec_a.script_name,
                script_b=spec_b.script_name,
                compatible=False,
                compatibility_issues=[f"Logical matching test failed: {str(e)}"],
                execution_time=time.time() - start_time,
            )

    def _create_file_mapping(
        self, path_matches: List[PathMatch], output_files: List[Path]
    ) -> Dict[str, str]:
        """
        Create mapping from logical names to actual file paths

        Args:
            path_matches: List of successful path matches
            output_files: List of actual output files

        Returns:
            Dictionary mapping destination logical names to file paths
        """
        file_mapping = {}

        # Simple heuristic: map files to logical names based on filename similarity
        for match in path_matches:
            best_file = self._find_best_file_for_logical_name(
                match.matched_source_name, output_files
            )
            if best_file:
                file_mapping[match.dest_logical_name] = str(best_file)

        return file_mapping

    def _find_best_file_for_logical_name(
        self, logical_name: str, output_files: List[Path]
    ) -> Optional[Path]:
        """
        Find the best matching file for a logical name

        Args:
            logical_name: Logical name to match
            output_files: List of available output files

        Returns:
            Best matching file path or None
        """
        if not output_files:
            return None

        # Simple heuristic: look for filename containing logical name
        logical_name_lower = logical_name.lower()

        # First, try exact matches in filename
        for file_path in output_files:
            if logical_name_lower in file_path.stem.lower():
                return file_path

        # If no match found, return the most recently modified file
        return max(output_files, key=lambda f: f.stat().st_mtime)

    def _detect_primary_format(self, files: List[Path]) -> str:
        """
        Detect the primary file format from a list of files

        Args:
            files: List of file paths

        Returns:
            Primary file format (extension)
        """
        if not files:
            return "unknown"

        # Count extensions
        extensions = {}
        for file_path in files:
            ext = file_path.suffix.lower() or "no_extension"
            extensions[ext] = extensions.get(ext, 0) + 1

        # Return most common extension
        return max(extensions.items(), key=lambda x: x[1])[0]

    def test_pipeline_with_topological_execution(
        self,
        dag,
        script_specs: Dict[str, EnhancedScriptExecutionSpec],
        script_tester_func,
    ) -> Dict[str, Any]:
        """
        Test pipeline with proper topological execution order

        Args:
            dag: PipelineDAG object
            script_specs: Dictionary of enhanced script specifications
            script_tester_func: Function to test individual scripts

        Returns:
            Dictionary with comprehensive pipeline test results
        """
        results = {
            "pipeline_success": True,
            "script_results": {},
            "data_flow_results": {},
            "execution_order": [],
            "logical_matching_results": {},
            "errors": [],
        }

        try:
            # Validate DAG structure
            validation_errors = self.topological_executor.validate_dag_structure(
                dag, script_specs
            )
            if validation_errors:
                results["pipeline_success"] = False
                results["errors"].extend(validation_errors)
                return results

            # Get topological execution order
            try:
                execution_order = self.topological_executor.get_execution_order(dag)
                results["execution_order"] = execution_order
            except ValueError as e:
                results["pipeline_success"] = False
                results["errors"].append(str(e))
                return results

            # Execute scripts in topological order
            executed_nodes = set()

            for current_node in execution_order:
                if current_node not in script_specs:
                    results["pipeline_success"] = False
                    results["errors"].append(
                        f"No ScriptExecutionSpec found for node: {current_node}"
                    )
                    continue

                # Test individual script functionality
                script_spec = script_specs[current_node]
                script_result = script_tester_func(script_spec)
                results["script_results"][current_node] = script_result

                if not script_result.success:
                    results["pipeline_success"] = False
                    results["errors"].append(
                        f"Script {current_node} failed: {script_result.error_message}"
                    )
                    continue

                executed_nodes.add(current_node)

                # Test logical name matching with dependent nodes
                outgoing_edges = [
                    (src, dst) for src, dst in dag.edges if src == current_node
                ]

                for src_node, dst_node in outgoing_edges:
                    if dst_node not in script_specs:
                        results["pipeline_success"] = False
                        results["errors"].append(
                            f"Missing ScriptExecutionSpec for destination node: {dst_node}"
                        )
                        continue

                    # Test logical name matching (without actual execution)
                    spec_a = script_specs[src_node]
                    spec_b = script_specs[dst_node]

                    path_matches = self.path_matcher.find_path_matches(spec_a, spec_b)
                    matching_details = self.path_matcher.generate_matching_report(
                        path_matches
                    )

                    edge_key = f"{src_node}->{dst_node}"
                    results["logical_matching_results"][edge_key] = {
                        "path_matches": len(path_matches),
                        "high_confidence_matches": len(
                            [m for m in path_matches if m.confidence >= 0.8]
                        ),
                        "matching_details": matching_details,
                    }

                    if not path_matches:
                        results["pipeline_success"] = False
                        results["errors"].append(
                            f"No logical name matches found for edge: {edge_key}"
                        )

            # Validate all edges were processed
            expected_edges = set(f"{src}->{dst}" for src, dst in dag.edges)
            processed_edges = set(results["logical_matching_results"].keys())
            missing_edges = expected_edges - processed_edges

            if missing_edges:
                results["pipeline_success"] = False
                results["errors"].append(
                    f"Unprocessed edges: {', '.join(missing_edges)}"
                )

            return results

        except Exception as e:
            results["pipeline_success"] = False
            results["errors"].append(
                f"Pipeline topological execution test failed: {str(e)}"
            )
            return results

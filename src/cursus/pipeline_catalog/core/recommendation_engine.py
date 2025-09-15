"""
Pipeline Recommendation Engine

Intelligent pipeline recommendation combining Zettelkasten principles.
Integrates manual linking (connections) with emergent organization (tags)
to provide contextual pipeline recommendations.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, Counter
from pydantic import BaseModel

from .catalog_registry import CatalogRegistry
from .connection_traverser import ConnectionTraverser, PipelineConnection
from .tag_discovery import TagBasedDiscovery

logger = logging.getLogger(__name__)


class RecommendationResult(BaseModel):
    """Result from recommendation engine."""

    pipeline_id: str
    title: str
    score: float
    reasoning: str
    connection_path: Optional[List[str]] = None
    tag_overlap: Optional[float] = None
    framework: Optional[str] = None
    complexity: Optional[str] = None


class CompositionRecommendation(BaseModel):
    """Recommendation for pipeline composition."""

    pipeline_sequence: List[str]
    composition_type: str  # sequential, parallel, conditional
    description: str
    estimated_complexity: str
    total_score: float


class PipelineRecommendationEngine:
    """
    Intelligent pipeline recommendation combining Zettelkasten principles.

    Integrates manual linking (connections) with emergent organization (tags)
    to provide contextual pipeline recommendations.
    """

    def __init__(
        self,
        registry: CatalogRegistry,
        traverser: ConnectionTraverser,
        discovery: TagBasedDiscovery,
    ):
        """
        Initialize with utility instances.

        Args:
            registry: CatalogRegistry instance
            traverser: ConnectionTraverser instance
            discovery: TagBasedDiscovery instance
        """
        self.registry = registry
        self.traverser = traverser
        self.discovery = discovery
        self._recommendation_cache = {}
        self._cache_valid = False

    def recommend_for_use_case(
        self, use_case: str, constraints: Optional[Dict[str, Any]] = None
    ) -> List[RecommendationResult]:
        """
        Recommend pipelines for specific use case.

        Args:
            use_case: Use case description or keywords
            constraints: Optional constraints (framework, complexity, etc.)

        Returns:
            List of recommendation results sorted by relevance
        """
        try:
            if constraints is None:
                constraints = {}

            recommendations = []

            # Text-based search for use case
            text_results = self.discovery.search_by_text(
                use_case, search_fields=["all"]
            )

            # Tag-based search for use case keywords
            use_case_words = use_case.lower().split()
            tag_results = []

            for word in use_case_words:
                word_results = self.discovery.find_by_tags([word], match_mode="any")
                tag_results.extend(word_results)

            # Combine and score results
            all_candidates = set()
            candidate_scores = defaultdict(float)
            candidate_reasons = defaultdict(list)

            # Score text search results
            for pipeline_id, text_score in text_results:
                all_candidates.add(pipeline_id)
                candidate_scores[pipeline_id] += (
                    text_score * 0.6
                )  # 60% weight for text relevance
                candidate_reasons[pipeline_id].append(
                    f"Text relevance: {text_score:.2f}"
                )

            # Score tag-based results
            tag_counter = Counter(tag_results)
            for pipeline_id, tag_count in tag_counter.items():
                all_candidates.add(pipeline_id)
                tag_score = min(tag_count * 0.5, 2.0)  # Cap at 2.0
                candidate_scores[pipeline_id] += (
                    tag_score * 0.4
                )  # 40% weight for tag matches
                candidate_reasons[pipeline_id].append(f"Tag matches: {tag_count}")

            # Apply constraints and build recommendations
            for pipeline_id in all_candidates:
                if self._meets_constraints(pipeline_id, constraints):
                    node = self.registry.get_pipeline_node(pipeline_id)
                    if node:
                        recommendations.append(
                            RecommendationResult(
                                pipeline_id=pipeline_id,
                                title=node.get("title", pipeline_id),
                                score=candidate_scores[pipeline_id],
                                reasoning="; ".join(candidate_reasons[pipeline_id]),
                                framework=node.get("zettelkasten_metadata", {}).get(
                                    "framework"
                                ),
                                complexity=node.get("zettelkasten_metadata", {}).get(
                                    "complexity"
                                ),
                            )
                        )

            # Sort by score (descending)
            recommendations.sort(key=lambda x: x.score, reverse=True)

            # Return top 10 recommendations
            return recommendations[:10]

        except Exception as e:
            logger.error(f"Failed to recommend for use case '{use_case}': {e}")
            return []

    def recommend_next_steps(self, current_pipeline: str) -> List[RecommendationResult]:
        """
        Recommend logical next steps after current pipeline.

        Args:
            current_pipeline: Current pipeline identifier

        Returns:
            List of recommendation results for next steps
        """
        try:
            recommendations = []

            # Get direct connections (manual linking)
            connections = self.traverser.get_all_connections(current_pipeline)

            # Prioritize "used_in" connections as natural next steps
            used_in_connections = connections.get("used_in", [])
            for conn in used_in_connections:
                node = self.registry.get_pipeline_node(conn.target_id)
                if node:
                    recommendations.append(
                        RecommendationResult(
                            pipeline_id=conn.target_id,
                            title=node.get("title", conn.target_id),
                            score=3.0,  # High score for direct "used_in" connections
                            reasoning=f"Direct composition: {conn.annotation}",
                            connection_path=[current_pipeline, conn.target_id],
                            framework=node.get("zettelkasten_metadata", {}).get(
                                "framework"
                            ),
                            complexity=node.get("zettelkasten_metadata", {}).get(
                                "complexity"
                            ),
                        )
                    )

            # Look for related pipelines that might be logical next steps
            related_connections = connections.get("related", [])
            for conn in related_connections:
                node = self.registry.get_pipeline_node(conn.target_id)
                if node:
                    # Check if this is a logical progression (e.g., training -> evaluation)
                    current_node = self.registry.get_pipeline_node(current_pipeline)
                    if current_node and self._is_logical_progression(
                        current_node, node
                    ):
                        recommendations.append(
                            RecommendationResult(
                                pipeline_id=conn.target_id,
                                title=node.get("title", conn.target_id),
                                score=2.5,  # Good score for logical progressions
                                reasoning=f"Logical progression: {conn.annotation}",
                                connection_path=[current_pipeline, conn.target_id],
                                framework=node.get("zettelkasten_metadata", {}).get(
                                    "framework"
                                ),
                                complexity=node.get("zettelkasten_metadata", {}).get(
                                    "complexity"
                                ),
                            )
                        )

            # Find similar pipelines with higher complexity (natural progression)
            similar_pipelines = self.discovery.suggest_similar_pipelines(
                current_pipeline, similarity_threshold=0.4
            )
            current_node = self.registry.get_pipeline_node(current_pipeline)
            current_complexity = (
                current_node.get("zettelkasten_metadata", {}).get(
                    "complexity", "simple"
                )
                if current_node
                else "simple"
            )

            complexity_order = {
                "simple": 0,
                "standard": 1,
                "advanced": 2,
                "comprehensive": 3,
            }
            current_complexity_level = complexity_order.get(current_complexity, 0)

            for similar_id, similarity in similar_pipelines:
                if similar_id in [r.pipeline_id for r in recommendations]:
                    continue  # Skip if already recommended

                similar_node = self.registry.get_pipeline_node(similar_id)
                if similar_node:
                    similar_complexity = similar_node.get(
                        "zettelkasten_metadata", {}
                    ).get("complexity", "simple")
                    similar_complexity_level = complexity_order.get(
                        similar_complexity, 0
                    )

                    # Recommend if it's more complex (natural progression)
                    if similar_complexity_level > current_complexity_level:
                        score = 2.0 + similarity  # Base score + similarity bonus
                        recommendations.append(
                            RecommendationResult(
                                pipeline_id=similar_id,
                                title=similar_node.get("title", similar_id),
                                score=score,
                                reasoning=f"Complexity progression: {current_complexity} -> {similar_complexity} (similarity: {similarity:.2f})",
                                tag_overlap=similarity,
                                framework=similar_node.get(
                                    "zettelkasten_metadata", {}
                                ).get("framework"),
                                complexity=similar_complexity,
                            )
                        )

            # Sort by score (descending)
            recommendations.sort(key=lambda x: x.score, reverse=True)

            # Return top 8 recommendations
            return recommendations[:8]

        except Exception as e:
            logger.error(f"Failed to recommend next steps for {current_pipeline}: {e}")
            return []

    def recommend_alternatives(
        self, current_pipeline: str, reason: str = "general"
    ) -> List[RecommendationResult]:
        """
        Recommend alternative approaches to current pipeline.

        Args:
            current_pipeline: Current pipeline identifier
            reason: Reason for seeking alternatives ('performance', 'simplicity', 'features', 'general')

        Returns:
            List of alternative recommendations
        """
        try:
            recommendations = []

            # Get direct alternatives (manual linking)
            alternatives = self.traverser.get_alternatives(current_pipeline)
            for alt in alternatives:
                node = self.registry.get_pipeline_node(alt.target_id)
                if node:
                    score = 3.0  # High score for curated alternatives

                    # Adjust score based on reason
                    if reason == "simplicity":
                        current_node = self.registry.get_pipeline_node(current_pipeline)
                        if current_node and self._is_simpler(node, current_node):
                            score += 0.5
                    elif reason == "performance":
                        if (
                            "performance" in alt.annotation.lower()
                            or "fast" in alt.annotation.lower()
                        ):
                            score += 0.5
                    elif reason == "features":
                        if (
                            "feature" in alt.annotation.lower()
                            or "enhanced" in alt.annotation.lower()
                        ):
                            score += 0.5

                    recommendations.append(
                        RecommendationResult(
                            pipeline_id=alt.target_id,
                            title=node.get("title", alt.target_id),
                            score=score,
                            reasoning=f"Curated alternative: {alt.annotation}",
                            connection_path=[current_pipeline, alt.target_id],
                            framework=node.get("zettelkasten_metadata", {}).get(
                                "framework"
                            ),
                            complexity=node.get("zettelkasten_metadata", {}).get(
                                "complexity"
                            ),
                        )
                    )

            # Find similar pipelines with different frameworks (alternative approaches)
            similar_pipelines = self.discovery.suggest_similar_pipelines(
                current_pipeline, similarity_threshold=0.3
            )
            current_node = self.registry.get_pipeline_node(current_pipeline)
            current_framework = (
                current_node.get("zettelkasten_metadata", {}).get("framework")
                if current_node
                else None
            )

            for similar_id, similarity in similar_pipelines:
                if similar_id in [r.pipeline_id for r in recommendations]:
                    continue  # Skip if already recommended

                similar_node = self.registry.get_pipeline_node(similar_id)
                if similar_node:
                    similar_framework = similar_node.get(
                        "zettelkasten_metadata", {}
                    ).get("framework")

                    # Recommend if it's a different framework (alternative approach)
                    if similar_framework != current_framework:
                        score = 2.0 + similarity * 0.5  # Base score + similarity bonus

                        # Adjust score based on reason
                        if reason == "simplicity" and self._is_simpler(
                            similar_node, current_node
                        ):
                            score += 0.5

                        recommendations.append(
                            RecommendationResult(
                                pipeline_id=similar_id,
                                title=similar_node.get("title", similar_id),
                                score=score,
                                reasoning=f"Alternative framework: {current_framework} -> {similar_framework} (similarity: {similarity:.2f})",
                                tag_overlap=similarity,
                                framework=similar_framework,
                                complexity=similar_node.get(
                                    "zettelkasten_metadata", {}
                                ).get("complexity"),
                            )
                        )

            # Sort by score (descending)
            recommendations.sort(key=lambda x: x.score, reverse=True)

            # Return top 6 recommendations
            return recommendations[:6]

        except Exception as e:
            logger.error(
                f"Failed to recommend alternatives for {current_pipeline}: {e}"
            )
            return []

    def recommend_compositions(
        self, pipeline_ids: List[str]
    ) -> List[CompositionRecommendation]:
        """
        Recommend ways to compose multiple pipelines.

        Args:
            pipeline_ids: List of pipeline identifiers to compose

        Returns:
            List of composition recommendations
        """
        try:
            if len(pipeline_ids) < 2:
                return []

            compositions = []

            # Analyze pipeline characteristics
            pipeline_info = {}
            for pid in pipeline_ids:
                node = self.registry.get_pipeline_node(pid)
                if node:
                    pipeline_info[pid] = {
                        "node": node,
                        "framework": node.get("zettelkasten_metadata", {}).get(
                            "framework"
                        ),
                        "complexity": node.get("zettelkasten_metadata", {}).get(
                            "complexity"
                        ),
                        "task_tags": node.get("multi_dimensional_tags", {}).get(
                            "task_tags", []
                        ),
                        "pattern_tags": node.get("multi_dimensional_tags", {}).get(
                            "pattern_tags", []
                        ),
                    }

            # Sequential composition recommendations
            sequential_compositions = self._recommend_sequential_compositions(
                pipeline_ids, pipeline_info
            )
            compositions.extend(sequential_compositions)

            # Parallel composition recommendations
            parallel_compositions = self._recommend_parallel_compositions(
                pipeline_ids, pipeline_info
            )
            compositions.extend(parallel_compositions)

            # Conditional composition recommendations
            conditional_compositions = self._recommend_conditional_compositions(
                pipeline_ids, pipeline_info
            )
            compositions.extend(conditional_compositions)

            # Sort by total score (descending)
            compositions.sort(key=lambda x: x.total_score, reverse=True)

            # Return top 5 compositions
            return compositions[:5]

        except Exception as e:
            logger.error(f"Failed to recommend compositions for {pipeline_ids}: {e}")
            return []

    def get_learning_path(
        self, start_complexity: str = "simple", target_framework: str = "any"
    ) -> List[str]:
        """
        Get learning path from simple to complex pipelines.

        Args:
            start_complexity: Starting complexity level
            target_framework: Target framework or 'any'

        Returns:
            Ordered list of pipeline IDs representing learning path
        """
        try:
            complexity_order = ["simple", "standard", "advanced", "comprehensive"]

            if start_complexity not in complexity_order:
                start_complexity = "simple"

            start_index = complexity_order.index(start_complexity)
            learning_path = []

            # Build path through complexity levels
            for i in range(start_index, len(complexity_order)):
                complexity = complexity_order[i]

                # Find pipelines at this complexity level
                if target_framework == "any":
                    candidates = self.discovery.find_by_complexity(complexity)
                else:
                    # Find pipelines matching both complexity and framework
                    complexity_pipelines = set(
                        self.discovery.find_by_complexity(complexity)
                    )
                    framework_pipelines = set(
                        self.discovery.find_by_framework(target_framework)
                    )
                    candidates = list(complexity_pipelines & framework_pipelines)

                if candidates:
                    # Select the best candidate for this level
                    best_candidate = self._select_best_learning_candidate(
                        candidates, learning_path[-1] if learning_path else None
                    )
                    if best_candidate:
                        learning_path.append(best_candidate)

            return learning_path

        except Exception as e:
            logger.error(f"Failed to get learning path: {e}")
            return []

    def _meets_constraints(self, pipeline_id: str, constraints: Dict[str, Any]) -> bool:
        """Check if pipeline meets specified constraints."""
        try:
            node = self.registry.get_pipeline_node(pipeline_id)
            if not node:
                return False

            metadata = node.get("zettelkasten_metadata", {})

            # Check framework constraint
            if "framework" in constraints:
                if metadata.get("framework") != constraints["framework"]:
                    return False

            # Check complexity constraint
            if "complexity" in constraints:
                if metadata.get("complexity") != constraints["complexity"]:
                    return False

            # Check skill level constraint
            if "skill_level" in constraints:
                discovery_meta = node.get("discovery_metadata", {})
                if discovery_meta.get("skill_level") != constraints["skill_level"]:
                    return False

            # Check resource requirements constraint
            if "max_resources" in constraints:
                discovery_meta = node.get("discovery_metadata", {})
                resource_level = discovery_meta.get("resource_requirements", "unknown")
                max_allowed = constraints["max_resources"]

                resource_order = {"low": 0, "medium": 1, "high": 2, "very_high": 3}
                if resource_level in resource_order and max_allowed in resource_order:
                    if resource_order[resource_level] > resource_order[max_allowed]:
                        return False

            return True

        except Exception as e:
            logger.error(f"Failed to check constraints for {pipeline_id}: {e}")
            return False

    def _is_logical_progression(
        self, current_node: Dict[str, Any], next_node: Dict[str, Any]
    ) -> bool:
        """Check if next_node is a logical progression from current_node."""
        try:
            current_tasks = set(
                current_node.get("multi_dimensional_tags", {}).get("task_tags", [])
            )
            next_tasks = set(
                next_node.get("multi_dimensional_tags", {}).get("task_tags", [])
            )

            # Define logical progressions
            progressions = {
                "training": ["evaluation", "registration", "deployment"],
                "preprocessing": ["training", "evaluation"],
                "evaluation": ["registration", "deployment"],
                "registration": ["deployment"],
            }

            for current_task in current_tasks:
                if current_task in progressions:
                    for next_task in progressions[current_task]:
                        if next_task in next_tasks:
                            return True

            return False

        except Exception as e:
            logger.error(f"Failed to check logical progression: {e}")
            return False

    def _is_simpler(self, node1: Dict[str, Any], node2: Dict[str, Any]) -> bool:
        """Check if node1 is simpler than node2."""
        try:
            complexity_order = {
                "simple": 0,
                "standard": 1,
                "advanced": 2,
                "comprehensive": 3,
            }

            complexity1 = node1.get("zettelkasten_metadata", {}).get(
                "complexity", "simple"
            )
            complexity2 = node2.get("zettelkasten_metadata", {}).get(
                "complexity", "simple"
            )

            level1 = complexity_order.get(complexity1, 0)
            level2 = complexity_order.get(complexity2, 0)

            return level1 < level2

        except Exception as e:
            logger.error(f"Failed to compare complexity: {e}")
            return False

    def _recommend_sequential_compositions(
        self, pipeline_ids: List[str], pipeline_info: Dict[str, Dict[str, Any]]
    ) -> List[CompositionRecommendation]:
        """Recommend sequential compositions."""
        compositions = []

        try:
            # Try different orderings based on logical flow
            task_order = [
                "preprocessing",
                "training",
                "evaluation",
                "registration",
                "deployment",
            ]

            # Sort pipelines by task order
            def get_task_priority(pid):
                tasks = pipeline_info.get(pid, {}).get("task_tags", [])
                for i, task in enumerate(task_order):
                    if task in tasks:
                        return i
                return len(task_order)

            sorted_pipelines = sorted(pipeline_ids, key=get_task_priority)

            # Create sequential composition
            if len(sorted_pipelines) >= 2:
                score = 3.0  # Base score for logical ordering

                # Bonus for framework consistency
                frameworks = [
                    pipeline_info.get(pid, {}).get("framework")
                    for pid in sorted_pipelines
                ]
                if len(set(frameworks)) == 1:  # All same framework
                    score += 0.5

                # Estimate complexity
                complexities = [
                    pipeline_info.get(pid, {}).get("complexity", "simple")
                    for pid in sorted_pipelines
                ]
                max_complexity = max(
                    complexities,
                    key=lambda x: {
                        "simple": 0,
                        "standard": 1,
                        "advanced": 2,
                        "comprehensive": 3,
                    }.get(x, 0),
                )

                compositions.append(
                    CompositionRecommendation(
                        pipeline_sequence=sorted_pipelines,
                        composition_type="sequential",
                        description=f"Sequential ML workflow: {' -> '.join([t.split('_')[0] for t in sorted_pipelines])}",
                        estimated_complexity=max_complexity,
                        total_score=score,
                    )
                )

        except Exception as e:
            logger.error(f"Failed to recommend sequential compositions: {e}")

        return compositions

    def _recommend_parallel_compositions(
        self, pipeline_ids: List[str], pipeline_info: Dict[str, Dict[str, Any]]
    ) -> List[CompositionRecommendation]:
        """Recommend parallel compositions."""
        compositions = []

        try:
            # Group by task type for parallel execution
            task_groups = defaultdict(list)

            for pid in pipeline_ids:
                tasks = pipeline_info.get(pid, {}).get("task_tags", [])
                for task in tasks:
                    task_groups[task].append(pid)

            # Find groups with multiple pipelines (parallel candidates)
            for task, pids in task_groups.items():
                if len(pids) >= 2:
                    score = 2.5  # Base score for parallel execution

                    # Bonus for different frameworks (ensemble approach)
                    frameworks = [
                        pipeline_info.get(pid, {}).get("framework") for pid in pids
                    ]
                    if len(set(frameworks)) > 1:
                        score += 0.5

                    # Estimate complexity
                    complexities = [
                        pipeline_info.get(pid, {}).get("complexity", "simple")
                        for pid in pids
                    ]
                    max_complexity = max(
                        complexities,
                        key=lambda x: {
                            "simple": 0,
                            "standard": 1,
                            "advanced": 2,
                            "comprehensive": 3,
                        }.get(x, 0),
                    )

                    compositions.append(
                        CompositionRecommendation(
                            pipeline_sequence=pids,
                            composition_type="parallel",
                            description=f"Parallel {task} ensemble: {', '.join([f.title() for f in set(frameworks)])}",
                            estimated_complexity=max_complexity,
                            total_score=score,
                        )
                    )

        except Exception as e:
            logger.error(f"Failed to recommend parallel compositions: {e}")

        return compositions

    def _recommend_conditional_compositions(
        self, pipeline_ids: List[str], pipeline_info: Dict[str, Dict[str, Any]]
    ) -> List[CompositionRecommendation]:
        """Recommend conditional compositions."""
        compositions = []

        try:
            # Look for pipelines that could be conditional alternatives
            complexity_groups = defaultdict(list)

            for pid in pipeline_ids:
                complexity = pipeline_info.get(pid, {}).get("complexity", "simple")
                complexity_groups[complexity].append(pid)

            # Create conditional composition if we have different complexity levels
            if len(complexity_groups) >= 2:
                all_pids = []
                for complexity in ["simple", "standard", "advanced", "comprehensive"]:
                    if complexity in complexity_groups:
                        all_pids.extend(complexity_groups[complexity])

                if len(all_pids) >= 2:
                    score = 2.0  # Base score for conditional logic

                    compositions.append(
                        CompositionRecommendation(
                            pipeline_sequence=all_pids,
                            composition_type="conditional",
                            description=f"Adaptive complexity: Start simple, escalate if needed",
                            estimated_complexity="adaptive",
                            total_score=score,
                        )
                    )

        except Exception as e:
            logger.error(f"Failed to recommend conditional compositions: {e}")

        return compositions

    def _select_best_learning_candidate(
        self, candidates: List[str], previous_pipeline: Optional[str]
    ) -> Optional[str]:
        """Select the best candidate for learning path."""
        try:
            if not candidates:
                return None

            if len(candidates) == 1:
                return candidates[0]

            # Score candidates
            candidate_scores = {}

            for candidate in candidates:
                score = 0.0
                node = self.registry.get_pipeline_node(candidate)

                if not node:
                    continue

                # Prefer pipelines with good documentation
                if node.get("description"):
                    score += 1.0

                # Prefer pipelines with clear single responsibility
                atomic_props = node.get("atomic_properties", {})
                responsibility = atomic_props.get("single_responsibility", "")
                if responsibility and len(responsibility.split()) <= 10:
                    score += 0.5

                # Prefer pipelines marked as beginner-friendly
                complexity_tags = node.get("multi_dimensional_tags", {}).get(
                    "complexity_tags", []
                )
                if "beginner_friendly" in complexity_tags:
                    score += 0.5

                # If there's a previous pipeline, prefer related ones
                if previous_pipeline:
                    connections = self.traverser.get_all_connections(previous_pipeline)
                    all_connected = []
                    for conn_list in connections.values():
                        all_connected.extend([c.target_id for c in conn_list])

                    if candidate in all_connected:
                        score += 1.0

                candidate_scores[candidate] = score

            # Return candidate with highest score
            if candidate_scores:
                return max(candidate_scores.items(), key=lambda x: x[1])[0]

            return candidates[0]  # Fallback to first candidate

        except Exception as e:
            logger.error(f"Failed to select best learning candidate: {e}")
            return candidates[0] if candidates else None

    def clear_cache(self) -> None:
        """Clear the internal recommendation cache."""
        self._recommendation_cache = {}
        self._cache_valid = False
        logger.debug("Recommendation engine cache cleared")

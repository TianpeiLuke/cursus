"""
Tag-Based Discovery Utilities

Tag-based pipeline discovery implementing Zettelkasten anti-categories principle.
Enables emergent organization through multi-dimensional tagging rather than
rigid hierarchical categories.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import defaultdict, Counter
import re

from .catalog_registry import CatalogRegistry

logger = logging.getLogger(__name__)


class TagBasedDiscovery:
    """
    Tag-based pipeline discovery implementing Zettelkasten anti-categories principle.

    Enables emergent organization through multi-dimensional tagging rather than
    rigid hierarchical categories.
    """

    def __init__(self, registry: CatalogRegistry):
        """
        Initialize with registry instance.

        Args:
            registry: CatalogRegistry instance for accessing pipeline data
        """
        self.registry = registry
        self._tag_cache = {}
        self._cache_valid = False

    def find_by_tags(self, tags: List[str], match_mode: str = "any") -> List[str]:
        """
        Find pipelines matching specified tags.

        Args:
            tags: List of tags to search for
            match_mode: 'any' (OR), 'all' (AND), or 'exact' (exact match)

        Returns:
            List of matching pipeline identifiers
        """
        try:
            if not tags:
                return []

            # Get tag index
            tag_index = self._get_tag_index()

            # Collect pipelines for each tag
            tag_pipelines = []
            for tag in tags:
                matching_pipelines = set()

                # Search across all tag categories
                for category, category_tags in tag_index.items():
                    if tag in category_tags:
                        matching_pipelines.update(category_tags[tag])

                tag_pipelines.append(matching_pipelines)

            # Apply match mode
            if match_mode == "any":
                # Union of all sets
                result = set()
                for pipeline_set in tag_pipelines:
                    result.update(pipeline_set)
                return list(result)

            elif match_mode == "all":
                # Intersection of all sets
                if not tag_pipelines:
                    return []
                result = tag_pipelines[0]
                for pipeline_set in tag_pipelines[1:]:
                    result = result.intersection(pipeline_set)
                return list(result)

            elif match_mode == "exact":
                # Find pipelines that have exactly these tags (and no others)
                all_pipelines = self.registry.get_all_pipelines()
                exact_matches = []

                for pipeline_id in all_pipelines:
                    pipeline_tags = self._get_pipeline_tags(pipeline_id)
                    if set(pipeline_tags) == set(tags):
                        exact_matches.append(pipeline_id)

                return exact_matches

            else:
                logger.error(f"Invalid match mode: {match_mode}")
                return []

        except Exception as e:
            logger.error(f"Failed to find pipelines by tags {tags}: {e}")
            return []

    def find_by_framework(self, framework: str) -> List[str]:
        """
        Find pipelines for specific framework.

        Args:
            framework: Framework name (e.g., 'xgboost', 'pytorch')

        Returns:
            List of matching pipeline identifiers
        """
        try:
            tag_index = self._get_tag_index()
            framework_tags = tag_index.get("framework_tags", {})
            return framework_tags.get(framework, [])

        except Exception as e:
            logger.error(f"Failed to find pipelines by framework {framework}: {e}")
            return []

    def find_by_complexity(self, complexity: str) -> List[str]:
        """
        Find pipelines by complexity level.

        Args:
            complexity: Complexity level (e.g., 'simple', 'standard', 'advanced')

        Returns:
            List of matching pipeline identifiers
        """
        try:
            tag_index = self._get_tag_index()
            complexity_tags = tag_index.get("complexity_tags", {})
            return complexity_tags.get(complexity, [])

        except Exception as e:
            logger.error(f"Failed to find pipelines by complexity {complexity}: {e}")
            return []

    def find_by_task(self, task: str) -> List[str]:
        """
        Find pipelines for specific task type.

        Args:
            task: Task type (e.g., 'training', 'evaluation', 'preprocessing')

        Returns:
            List of matching pipeline identifiers
        """
        try:
            tag_index = self._get_tag_index()
            task_tags = tag_index.get("task_tags", {})
            return task_tags.get(task, [])

        except Exception as e:
            logger.error(f"Failed to find pipelines by task {task}: {e}")
            return []

    def find_by_domain(self, domain: str) -> List[str]:
        """
        Find pipelines for specific domain.

        Args:
            domain: Domain type (e.g., 'tabular', 'nlp', 'computer_vision')

        Returns:
            List of matching pipeline identifiers
        """
        try:
            tag_index = self._get_tag_index()
            domain_tags = tag_index.get("domain_tags", {})
            return domain_tags.get(domain, [])

        except Exception as e:
            logger.error(f"Failed to find pipelines by domain {domain}: {e}")
            return []

    def find_by_pattern(self, pattern: str) -> List[str]:
        """
        Find pipelines by architectural pattern.

        Args:
            pattern: Pattern type (e.g., 'atomic_workflow', 'end_to_end', 'modular')

        Returns:
            List of matching pipeline identifiers
        """
        try:
            tag_index = self._get_tag_index()
            pattern_tags = tag_index.get("pattern_tags", {})
            return pattern_tags.get(pattern, [])

        except Exception as e:
            logger.error(f"Failed to find pipelines by pattern {pattern}: {e}")
            return []

    def find_by_multiple_criteria(
        self,
        framework: Optional[str] = None,
        complexity: Optional[str] = None,
        task: Optional[str] = None,
        domain: Optional[str] = None,
        pattern: Optional[str] = None,
        additional_tags: Optional[List[str]] = None,
    ) -> List[str]:
        """
        Find pipelines matching multiple criteria.

        Args:
            framework: Framework filter
            complexity: Complexity filter
            task: Task filter
            domain: Domain filter
            pattern: Pattern filter
            additional_tags: Additional tags to match

        Returns:
            List of matching pipeline identifiers
        """
        try:
            # Collect all criteria
            criteria_tags = []

            if framework:
                criteria_tags.append(framework)
            if complexity:
                criteria_tags.append(complexity)
            if task:
                criteria_tags.append(task)
            if domain:
                criteria_tags.append(domain)
            if pattern:
                criteria_tags.append(pattern)
            if additional_tags:
                criteria_tags.extend(additional_tags)

            if not criteria_tags:
                return self.registry.get_all_pipelines()

            # Find pipelines matching all criteria
            return self.find_by_tags(criteria_tags, match_mode="all")

        except Exception as e:
            logger.error(f"Failed to find pipelines by multiple criteria: {e}")
            return []

    def search_by_text(
        self, query: str, search_fields: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Search pipelines by text query with relevance scoring.

        Args:
            query: Text query to search for
            search_fields: Fields to search in ('title', 'description', 'tags', 'all')

        Returns:
            List of (pipeline_id, relevance_score) tuples, sorted by relevance
        """
        try:
            if not query.strip():
                return []

            if search_fields is None:
                search_fields = ["all"]

            query_terms = [
                term.lower().strip() for term in re.split(r"\s+", query) if term.strip()
            ]
            if not query_terms:
                return []

            results = []
            all_pipelines = self.registry.get_all_pipelines()

            for pipeline_id in all_pipelines:
                node = self.registry.get_pipeline_node(pipeline_id)
                if not node:
                    continue

                score = 0.0

                # Search in title
                if "title" in search_fields or "all" in search_fields:
                    title = node.get("title", "").lower()
                    for term in query_terms:
                        if term in title:
                            score += 2.0  # Higher weight for title matches

                # Search in description
                if "description" in search_fields or "all" in search_fields:
                    description = node.get("description", "").lower()
                    for term in query_terms:
                        if term in description:
                            score += 1.0

                # Search in tags
                if "tags" in search_fields or "all" in search_fields:
                    all_tags = self._get_pipeline_tags(pipeline_id)
                    tag_text = " ".join(all_tags).lower()
                    for term in query_terms:
                        if term in tag_text:
                            score += 1.5  # Medium weight for tag matches

                # Search in atomic properties
                if "all" in search_fields:
                    atomic_props = node.get("atomic_properties", {})
                    responsibility = atomic_props.get(
                        "single_responsibility", ""
                    ).lower()
                    for term in query_terms:
                        if term in responsibility:
                            score += 1.2

                if score > 0:
                    results.append((pipeline_id, score))

            # Sort by relevance score (descending)
            results.sort(key=lambda x: x[1], reverse=True)
            return results

        except Exception as e:
            logger.error(f"Failed to search by text '{query}': {e}")
            return []

    def get_tag_clusters(self) -> Dict[str, List[str]]:
        """
        Get emergent clusters based on tag similarity.

        Returns:
            Dictionary mapping cluster names to pipeline lists
        """
        try:
            all_pipelines = self.registry.get_all_pipelines()

            # Build tag similarity matrix
            pipeline_tags = {}
            for pipeline_id in all_pipelines:
                pipeline_tags[pipeline_id] = set(self._get_pipeline_tags(pipeline_id))

            # Find clusters using tag overlap
            clusters = {}
            processed = set()

            for pipeline_id in all_pipelines:
                if pipeline_id in processed:
                    continue

                # Find similar pipelines
                similar_pipelines = [pipeline_id]
                pipeline_tag_set = pipeline_tags[pipeline_id]

                for other_id in all_pipelines:
                    if other_id != pipeline_id and other_id not in processed:
                        other_tag_set = pipeline_tags[other_id]

                        # Calculate Jaccard similarity
                        intersection = len(pipeline_tag_set & other_tag_set)
                        union = len(pipeline_tag_set | other_tag_set)

                        if union > 0:
                            similarity = intersection / union
                            if similarity >= 0.3:  # 30% similarity threshold
                                similar_pipelines.append(other_id)

                # Create cluster name from common tags
                if len(similar_pipelines) > 1:
                    common_tags = pipeline_tag_set
                    for other_id in similar_pipelines[1:]:
                        common_tags = common_tags & pipeline_tags[other_id]

                    cluster_name = (
                        "_".join(sorted(list(common_tags))[:3])
                        if common_tags
                        else f"cluster_{len(clusters)}"
                    )
                    clusters[cluster_name] = similar_pipelines

                    # Mark as processed
                    for pid in similar_pipelines:
                        processed.add(pid)
                else:
                    processed.add(pipeline_id)

            return clusters

        except Exception as e:
            logger.error(f"Failed to get tag clusters: {e}")
            return {}

    def suggest_similar_pipelines(
        self, pipeline_id: str, similarity_threshold: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Suggest similar pipelines based on tag overlap.

        Args:
            pipeline_id: Pipeline identifier to find similar pipelines for
            similarity_threshold: Minimum similarity score (0.0 to 1.0)

        Returns:
            List of (similar_pipeline_id, similarity_score) tuples
        """
        try:
            pipeline_tags = set(self._get_pipeline_tags(pipeline_id))
            if not pipeline_tags:
                return []

            similar_pipelines = []
            all_pipelines = self.registry.get_all_pipelines()

            for other_id in all_pipelines:
                if other_id == pipeline_id:
                    continue

                other_tags = set(self._get_pipeline_tags(other_id))
                if not other_tags:
                    continue

                # Calculate Jaccard similarity
                intersection = len(pipeline_tags & other_tags)
                union = len(pipeline_tags | other_tags)

                if union > 0:
                    similarity = intersection / union
                    if similarity >= similarity_threshold:
                        similar_pipelines.append((other_id, similarity))

            # Sort by similarity score (descending)
            similar_pipelines.sort(key=lambda x: x[1], reverse=True)
            return similar_pipelines

        except Exception as e:
            logger.error(f"Failed to suggest similar pipelines for {pipeline_id}: {e}")
            return []

    def get_tag_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about tag usage and distribution.

        Returns:
            Dictionary containing tag statistics
        """
        try:
            tag_index = self._get_tag_index()

            stats = {
                "total_tag_categories": len(tag_index),
                "tag_categories": {},
                "most_common_tags": {},
                "tag_distribution": {},
                "pipeline_tag_counts": {},
            }

            # Analyze each tag category
            for category, category_tags in tag_index.items():
                category_stats = {
                    "total_tags": len(category_tags),
                    "total_pipelines": sum(
                        len(pipelines) for pipelines in category_tags.values()
                    ),
                    "average_pipelines_per_tag": 0,
                    "most_common": [],
                }

                if category_tags:
                    # Calculate average
                    category_stats["average_pipelines_per_tag"] = category_stats[
                        "total_pipelines"
                    ] / len(category_tags)

                    # Find most common tags
                    tag_counts = [
                        (tag, len(pipelines))
                        for tag, pipelines in category_tags.items()
                    ]
                    tag_counts.sort(key=lambda x: x[1], reverse=True)
                    category_stats["most_common"] = tag_counts[:5]

                stats["tag_categories"][category] = category_stats

            # Overall most common tags across all categories
            all_tag_counts = Counter()
            for category_tags in tag_index.values():
                for tag, pipelines in category_tags.items():
                    all_tag_counts[tag] += len(pipelines)

            stats["most_common_tags"] = all_tag_counts.most_common(10)

            # Tag distribution analysis
            all_pipelines = self.registry.get_all_pipelines()
            tag_count_distribution = Counter()

            for pipeline_id in all_pipelines:
                pipeline_tags = self._get_pipeline_tags(pipeline_id)
                tag_count = len(pipeline_tags)
                tag_count_distribution[tag_count] += 1

            stats["tag_distribution"] = dict(tag_count_distribution)

            # Pipeline tag count statistics
            tag_counts = [len(self._get_pipeline_tags(pid)) for pid in all_pipelines]
            if tag_counts:
                stats["pipeline_tag_counts"] = {
                    "min": min(tag_counts),
                    "max": max(tag_counts),
                    "average": sum(tag_counts) / len(tag_counts),
                    "median": sorted(tag_counts)[len(tag_counts) // 2],
                }

            return stats

        except Exception as e:
            logger.error(f"Failed to get tag statistics: {e}")
            return {"error": str(e)}

    def find_undertagged_pipelines(self, min_tags: int = 3) -> List[Tuple[str, int]]:
        """
        Find pipelines with insufficient tags.

        Args:
            min_tags: Minimum number of tags expected

        Returns:
            List of (pipeline_id, tag_count) tuples for undertagged pipelines
        """
        try:
            undertagged = []
            all_pipelines = self.registry.get_all_pipelines()

            for pipeline_id in all_pipelines:
                pipeline_tags = self._get_pipeline_tags(pipeline_id)
                tag_count = len(pipeline_tags)

                if tag_count < min_tags:
                    undertagged.append((pipeline_id, tag_count))

            # Sort by tag count (ascending)
            undertagged.sort(key=lambda x: x[1])
            return undertagged

        except Exception as e:
            logger.error(f"Failed to find undertagged pipelines: {e}")
            return []

    def suggest_tags_for_pipeline(self, pipeline_id: str) -> Dict[str, List[str]]:
        """
        Suggest additional tags for a pipeline based on similar pipelines.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            Dictionary mapping tag categories to suggested tags
        """
        try:
            # Find similar pipelines
            similar_pipelines = self.suggest_similar_pipelines(
                pipeline_id, similarity_threshold=0.2
            )

            if not similar_pipelines:
                return {}

            # Get current tags
            current_tags = set(self._get_pipeline_tags(pipeline_id))

            # Collect tags from similar pipelines
            suggested_tags = defaultdict(Counter)

            for similar_id, similarity in similar_pipelines:
                similar_node = self.registry.get_pipeline_node(similar_id)
                if not similar_node:
                    continue

                similar_tag_dict = similar_node.get("multi_dimensional_tags", {})

                for category, tags in similar_tag_dict.items():
                    for tag in tags:
                        if tag not in current_tags:
                            # Weight by similarity score
                            suggested_tags[category][tag] += similarity

            # Convert to suggestions (top 3 per category)
            suggestions = {}
            for category, tag_counter in suggested_tags.items():
                top_tags = [tag for tag, score in tag_counter.most_common(3)]
                if top_tags:
                    suggestions[category] = top_tags

            return suggestions

        except Exception as e:
            logger.error(f"Failed to suggest tags for pipeline {pipeline_id}: {e}")
            return {}

    def _get_tag_index(self) -> Dict[str, Dict[str, List[str]]]:
        """Get the tag index from registry."""
        if not self._cache_valid:
            registry_data = self.registry.load_registry()
            self._tag_cache = registry_data.get("tag_index", {})
            self._cache_valid = True

        return self._tag_cache

    def _get_pipeline_tags(self, pipeline_id: str) -> List[str]:
        """Get all tags for a pipeline as a flat list."""
        node = self.registry.get_pipeline_node(pipeline_id)
        if not node:
            return []

        all_tags = []
        tag_dict = node.get("multi_dimensional_tags", {})

        for tag_list in tag_dict.values():
            all_tags.extend(tag_list)

        return list(set(all_tags))  # Remove duplicates

    def clear_cache(self) -> None:
        """Clear the internal tag cache."""
        self._tag_cache = {}
        self._cache_valid = False
        logger.debug("Tag discovery cache cleared")

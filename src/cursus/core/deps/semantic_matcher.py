"""
Semantic matching utilities for intelligent dependency resolution.

This module provides algorithms for calculating semantic similarity between
dependency names and output names to enable intelligent auto-resolution.
"""

import re
from typing import List, Set, Dict, Tuple, Any
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)


class SemanticMatcher:
    """Semantic similarity matching for dependency resolution."""

    def __init__(self) -> None:
        """Initialize the semantic matcher with common patterns."""
        # Common synonyms for pipeline concepts
        self.synonyms = {
            "model": ["model", "artifact", "trained", "output"],
            "data": ["data", "dataset", "input", "processed", "training"],
            "config": [
                "config",
                "configuration",
                "params",
                "parameters",
                "hyperparameters",
                "settings",
            ],
            "payload": ["payload", "sample", "test", "inference", "example"],
            "output": ["output", "result", "artifact", "generated", "produced"],
            "training": ["training", "train", "fit", "learn"],
            "preprocessing": [
                "preprocessing",
                "preprocess",
                "processed",
                "clean",
                "transform",
            ],
        }

        # Common abbreviations and expansions
        self.abbreviations = {
            "config": "configuration",
            "params": "parameters",
            "hyperparams": "hyperparameters",
            "preprocess": "preprocessing",
            "eval": "evaluation",
            "reg": "registration",
            "pkg": "package",
            "packaged": "package",
        }

        # Stop words that should be ignored in matching
        self.stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }

    def calculate_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate semantic similarity between two names.

        Args:
            name1: First name to compare
            name2: Second name to compare

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not name1 or not name2:
            return 0.0

        # Normalize names
        norm1 = self._normalize_name(name1)
        norm2 = self._normalize_name(name2)

        # Exact match after normalization
        if norm1 == norm2:
            return 1.0

        # Calculate multiple similarity metrics
        scores = []

        # 1. String similarity (30% weight)
        string_sim = self._calculate_string_similarity(norm1, norm2)
        scores.append(("string", string_sim, 0.3))

        # 2. Token overlap (25% weight)
        token_sim = self._calculate_token_similarity(norm1, norm2)
        scores.append(("token", token_sim, 0.25))

        # 3. Semantic similarity (25% weight)
        semantic_sim = self._calculate_semantic_similarity(norm1, norm2)
        scores.append(("semantic", semantic_sim, 0.25))

        # 4. Substring matching (20% weight)
        substring_sim = self._calculate_substring_similarity(norm1, norm2)
        scores.append(("substring", substring_sim, 0.2))

        # Calculate weighted average
        total_score = sum(score * weight for _, score, weight in scores)

        logger.debug(
            f"Similarity '{name1}' vs '{name2}': {total_score:.3f} "
            f"(details: {[(name, f'{score:.3f}') for name, score, _ in scores]})"
        )

        return total_score

    def calculate_similarity_with_aliases(self, name: str, output_spec: Any) -> float:
        """
        Calculate semantic similarity between a name and an output specification,
        considering both logical_name and all aliases.

        Args:
            name: The name to compare (typically the dependency's logical_name)
            output_spec: OutputSpec with logical_name and potential aliases

        Returns:
            The highest similarity score (0.0 to 1.0) between name and any name in output_spec
        """
        # Start with similarity to logical_name
        best_score = self.calculate_similarity(name, output_spec.logical_name)
        best_match = output_spec.logical_name

        # Check each alias
        for alias in output_spec.aliases:
            alias_score = self.calculate_similarity(name, alias)
            if alias_score > best_score:
                best_score = alias_score
                best_match = alias

        # Log which name gave the best match (only for meaningful matches)
        if best_score > 0.5:
            logger.debug(
                f"Best match for '{name}': '{best_match}' (score: {best_score:.3f})"
            )

        return best_score

    def _normalize_name(self, name: str) -> str:
        """Normalize a name for comparison."""
        # Convert to lowercase
        normalized = name.lower()

        # Remove common separators and replace with spaces
        normalized = re.sub(r"[_\-\.]", " ", normalized)

        # Remove special characters
        normalized = re.sub(r"[^a-z0-9\s]", "", normalized)

        # Expand abbreviations
        words = normalized.split()
        expanded_words = []
        for word in words:
            expanded = self.abbreviations.get(word, word)
            expanded_words.append(expanded)

        # Remove stop words
        filtered_words = [
            word for word in expanded_words if word not in self.stop_words
        ]

        return " ".join(filtered_words)

    def _calculate_string_similarity(self, name1: str, name2: str) -> float:
        """Calculate string similarity using sequence matching."""
        return SequenceMatcher(None, name1, name2).ratio()

    def _calculate_token_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity based on token overlap."""
        tokens1 = set(name1.split())
        tokens2 = set(name2.split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        return len(intersection) / len(union) if union else 0.0

    def _calculate_semantic_similarity(self, name1: str, name2: str) -> float:
        """Calculate semantic similarity using synonym matching."""
        tokens1 = set(name1.split())
        tokens2 = set(name2.split())

        # Find semantic matches
        semantic_matches: float = 0.0
        total_comparisons = 0

        for token1 in tokens1:
            for token2 in tokens2:
                total_comparisons += 1

                # Direct match
                if token1 == token2:
                    semantic_matches += 1
                    continue

                # Synonym match
                if self._are_synonyms(token1, token2):
                    semantic_matches += 0.8  # Slightly lower score for synonyms

        return semantic_matches / total_comparisons if total_comparisons > 0 else 0.0

    def _calculate_substring_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity based on substring matching."""
        # Check if one is a substring of the other
        if name1 in name2 or name2 in name1:
            shorter = min(len(name1), len(name2))
            longer = max(len(name1), len(name2))
            return shorter / longer

        # Check for common substrings
        words1 = name1.split()
        words2 = name2.split()

        max_substring_score = 0.0
        for word1 in words1:
            for word2 in words2:
                if (
                    len(word1) >= 3 and len(word2) >= 3
                ):  # Only consider meaningful substrings
                    if word1 in word2 or word2 in word1:
                        shorter = min(len(word1), len(word2))
                        longer = max(len(word1), len(word2))
                        score = shorter / longer
                        max_substring_score = max(max_substring_score, score)

        return max_substring_score

    def _are_synonyms(self, word1: str, word2: str) -> bool:
        """Check if two words are synonyms."""
        for concept, synonyms in self.synonyms.items():
            if word1 in synonyms and word2 in synonyms:
                return True
        return False

    def find_best_matches(
        self, target_name: str, candidate_names: List[str], threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Find the best matching names from a list of candidates.

        Args:
            target_name: Name to match against
            candidate_names: List of candidate names
            threshold: Minimum similarity threshold

        Returns:
            List of (name, score) tuples sorted by score (highest first)
        """
        matches = []

        for candidate in candidate_names:
            score = self.calculate_similarity(target_name, candidate)
            if score >= threshold:
                matches.append((candidate, score))

        # Sort by score (highest first)
        matches.sort(key=lambda x: x[1], reverse=True)

        return matches

    def explain_similarity(self, name1: str, name2: str) -> Dict[str, Any]:
        """
        Provide detailed explanation of similarity calculation.

        Args:
            name1: First name to compare
            name2: Second name to compare

        Returns:
            Dictionary with detailed similarity breakdown
        """
        norm1 = self._normalize_name(name1)
        norm2 = self._normalize_name(name2)

        explanation = {
            "overall_score": self.calculate_similarity(name1, name2),
            "normalized_names": (norm1, norm2),
            "string_similarity": self._calculate_string_similarity(norm1, norm2),
            "token_similarity": self._calculate_token_similarity(norm1, norm2),
            "semantic_similarity": self._calculate_semantic_similarity(norm1, norm2),
            "substring_similarity": self._calculate_substring_similarity(norm1, norm2),
        }

        return explanation

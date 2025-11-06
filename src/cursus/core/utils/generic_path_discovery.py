"""
Generic Path Discovery Algorithm for Project Folders.

This module implements a multi-directional recursive search algorithm to find
uniquely named project folders across different directory structures. It serves
as a robust fallback for path resolution when other strategies fail.

The algorithm searches both upward and downward from multiple reference points,
making it suitable for complex nested project structures.
"""

from typing import Optional, List, Tuple
from pathlib import Path
import logging
import time

logger = logging.getLogger(__name__)


def find_project_folder_generic(
    project_root_folder: str,
    max_depth_up: int = 5,
    max_depth_down: int = 3,
    reference_points: Optional[List[Path]] = None,
) -> Optional[Path]:
    """
    Generic algorithm to find a uniquely named project folder.

    Searches from multiple reference points in both upward and downward directions,
    using recursive traversal to handle nested directory structures.

    Args:
        project_root_folder: Name or path of project folder
                           (e.g., "atoz_xgboost" or "projects/atoz_xgboost")
        max_depth_up: Maximum levels to search upward (default: 5)
        max_depth_down: Maximum levels to search downward (default: 3)
        reference_points: Optional list of starting points for search.
                         If None, uses [Path.cwd(), cursus_package_root]

    Returns:
        Absolute path to project folder if found, None otherwise

    Example:
        >>> find_project_folder_generic("atoz_xgboost")
        PosixPath('/Users/user/workspace/projects/atoz_xgboost')

        >>> find_project_folder_generic("projects/atoz_xgboost")
        PosixPath('/Users/user/workspace/projects/atoz_xgboost')
    """
    start_time = time.time()

    try:
        # Parse project_root_folder to handle nested paths
        folder_parts = Path(project_root_folder).parts
        final_folder_name = folder_parts[-1]  # Last part is the actual folder

        # Determine reference points to start search from
        if reference_points is None:
            reference_points = _get_default_reference_points()

        logger.debug(
            f"Starting generic path discovery for '{project_root_folder}' "
            f"from {len(reference_points)} reference points"
        )

        # Try each reference point
        for ref_point in reference_points:
            logger.debug(f"Searching from reference point: {ref_point}")

            # Strategy 1: Search upward from reference point
            result = _search_upward(ref_point, project_root_folder, max_depth_up)
            if result:
                elapsed = time.time() - start_time
                logger.info(
                    f"Generic discovery succeeded via upward search: {result} "
                    f"(took {elapsed:.3f}s)"
                )
                return result

            # Strategy 2: Search downward from reference point
            result = _search_downward(ref_point, final_folder_name, max_depth_down)
            if result and _matches_full_path(result, folder_parts):
                elapsed = time.time() - start_time
                logger.info(
                    f"Generic discovery succeeded via downward search: {result} "
                    f"(took {elapsed:.3f}s)"
                )
                return result

        elapsed = time.time() - start_time
        logger.debug(
            f"Generic discovery failed for '{project_root_folder}' "
            f"(searched {len(reference_points)} reference points in {elapsed:.3f}s)"
        )
        return None

    except Exception as e:
        logger.warning(f"Generic path discovery error: {e}")
        return None


def _get_default_reference_points() -> List[Path]:
    """
    Get default reference points for path search.

    Returns:
        List of Path objects to use as starting points for search
    """
    reference_points = [Path.cwd()]  # Working directory always included

    # Try to add cursus package root if available
    try:
        # Navigate up from this file to find cursus package root
        cursus_file = Path(__file__)
        cursus_root = cursus_file.parent.parent.parent  # From utils -> core -> cursus
        if cursus_root.exists():
            reference_points.append(cursus_root)
    except Exception as e:
        logger.debug(f"Could not determine cursus package root: {e}")

    return reference_points


def _search_upward(
    start_path: Path, project_root_folder: str, max_depth: int
) -> Optional[Path]:
    """
    Search upward from start_path to find project_root_folder.

    Traverses parent directories up to max_depth levels, checking if
    project_root_folder exists at each level.

    Args:
        start_path: Starting directory for upward search
        project_root_folder: Folder name or path to find
        max_depth: Maximum number of parent levels to check

    Returns:
        Path to found folder, or None if not found
    """
    current = start_path
    depth = 0

    logger.debug(f"Upward search starting from: {current}")

    while current != current.parent and depth < max_depth:
        # Check if project_root_folder exists here
        candidate = current / project_root_folder

        logger.debug(f"  Checking (depth {depth}): {candidate}")

        if candidate.exists() and candidate.is_dir():
            logger.debug(f"  ✓ Found via upward search: {candidate}")
            return candidate

        # Move up one level
        current = current.parent
        depth += 1

    logger.debug(f"  ✗ Upward search found nothing (reached depth {depth})")
    return None


def _search_downward(
    start_path: Path, folder_name: str, max_depth: int, current_depth: int = 0
) -> Optional[Path]:
    """
    Recursively search downward from start_path to find folder_name.

    Uses depth-limited recursive traversal to search subdirectories.
    Implements early termination on first match.

    Args:
        start_path: Starting directory for downward search
        folder_name: Name of folder to find (not a path)
        max_depth: Maximum depth to recurse
        current_depth: Current recursion depth (internal use)

    Returns:
        Path to found folder, or None if not found
    """
    if current_depth >= max_depth:
        logger.debug(f"  Reached max depth {max_depth}, stopping recursion")
        return None

    if current_depth == 0:
        logger.debug(f"Downward search starting from: {start_path}")

    try:
        # Check immediate children
        for child in start_path.iterdir():
            if not child.is_dir():
                continue

            indent = "  " * (current_depth + 1)
            logger.debug(f"{indent}Checking: {child.name}")

            # Check if this child matches
            if child.name == folder_name:
                logger.debug(f"{indent}✓ Found via downward search: {child}")
                return child

            # Recursively search this child's subdirectories
            if current_depth + 1 < max_depth:
                result = _search_downward(
                    child, folder_name, max_depth, current_depth + 1
                )
                if result:
                    return result

    except PermissionError as e:
        # Skip directories we can't access
        logger.debug(f"  Permission denied, skipping: {start_path}")
    except Exception as e:
        # Log other errors but continue searching
        logger.debug(f"  Error scanning {start_path}: {e}")

    if current_depth == 0:
        logger.debug(f"  ✗ Downward search found nothing")

    return None


def _matches_full_path(found_path: Path, expected_parts: Tuple[str, ...]) -> bool:
    """
    Verify that found_path matches the full expected path structure.

    For example, if project_root_folder was "projects/atoz_xgboost",
    this verifies that the found path ends with "projects/atoz_xgboost".

    Args:
        found_path: Path that was found by search
        expected_parts: Tuple of path components to match

    Returns:
        True if found_path matches expected structure, False otherwise

    Example:
        >>> found = Path("/home/user/workspace/projects/atoz_xgboost")
        >>> expected = ("projects", "atoz_xgboost")
        >>> _matches_full_path(found, expected)
        True
    """
    if len(expected_parts) == 1:
        # Simple case: just a folder name, any match is valid
        return True

    # Complex case: nested path like "projects/atoz_xgboost"
    found_parts = found_path.parts

    # Check if the last N parts of found_path match expected_parts
    if len(found_parts) >= len(expected_parts):
        matches = found_parts[-len(expected_parts) :] == expected_parts
        logger.debug(
            f"Path structure match: {matches} "
            f"(found: {found_parts[-len(expected_parts) :]}, expected: {expected_parts})"
        )
        return matches

    logger.debug(
        f"Path structure mismatch: found path too short "
        f"(found: {len(found_parts)} parts, expected: {len(expected_parts)})"
    )
    return False


class GenericPathDiscoveryMetrics:
    """Track generic path discovery performance metrics."""

    def __init__(self):
        self.success_count = 0
        self.failure_count = 0
        self.total_attempts = 0
        self.search_times = []

    def record_success(self, search_time: float):
        """Record successful path discovery."""
        self.success_count += 1
        self.total_attempts += 1
        self.search_times.append(search_time)

    def record_failure(self, search_time: float):
        """Record failed path discovery."""
        self.failure_count += 1
        self.total_attempts += 1
        self.search_times.append(search_time)

    def get_metrics(self) -> dict:
        """Get current performance metrics."""
        if self.total_attempts == 0:
            return {"status": "no_data"}

        return {
            "success_rate": self.success_count / self.total_attempts,
            "failure_rate": self.failure_count / self.total_attempts,
            "average_search_time": sum(self.search_times) / len(self.search_times)
            if self.search_times
            else 0,
            "total_attempts": self.total_attempts,
        }


# Global metrics instance
_generic_discovery_metrics = GenericPathDiscoveryMetrics()


def get_generic_discovery_metrics() -> dict:
    """Get current generic path discovery performance metrics."""
    return _generic_discovery_metrics.get_metrics()

"""
Hybrid Path Resolution System for Cursus.

This module implements the hybrid strategy deployment path resolution system
that works across Lambda/MODS bundled, development monorepo, and pip-installed
separated deployment scenarios.

The hybrid approach uses two complementary strategies:
1. Package Location Discovery (Primary) - Uses Path(__file__) from cursus package
2. Working Directory Discovery (Fallback) - Uses Path.cwd() for traversal
"""

from typing import Dict, Any, Optional
from pathlib import Path
import logging
import time
import os

from .generic_path_discovery import find_project_folder_generic

logger = logging.getLogger(__name__)


class HybridResolutionMetrics:
    """Track hybrid resolution performance metrics."""

    def __init__(self):
        self.strategy_1_success_count = 0
        self.strategy_2_success_count = 0
        self.strategy_3_success_count = 0
        self.strategy_4_success_count = 0
        self.total_resolution_attempts = 0
        self.resolution_times = []
        self.failure_count = 0

    def record_strategy_1_success(self, resolution_time: float):
        """Record successful Package Location Discovery."""
        self.strategy_1_success_count += 1
        self.total_resolution_attempts += 1
        self.resolution_times.append(resolution_time)

    def record_strategy_2_success(self, resolution_time: float):
        """Record successful Working Directory Discovery."""
        self.strategy_2_success_count += 1
        self.total_resolution_attempts += 1
        self.resolution_times.append(resolution_time)

    def record_strategy_3_success(self, resolution_time: float):
        """Record successful Generic Path Discovery."""
        self.strategy_3_success_count += 1
        self.total_resolution_attempts += 1
        self.resolution_times.append(resolution_time)

    def record_strategy_4_success(self, resolution_time: float):
        """Record successful Default Scripts Discovery."""
        self.strategy_4_success_count += 1
        self.total_resolution_attempts += 1
        self.resolution_times.append(resolution_time)

    def record_failure(self, resolution_time: float):
        """Record resolution failure."""
        self.failure_count += 1
        self.total_resolution_attempts += 1
        self.resolution_times.append(resolution_time)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        if self.total_resolution_attempts == 0:
            return {"status": "no_data"}

        return {
            "strategy_1_success_rate": self.strategy_1_success_count
            / self.total_resolution_attempts,
            "strategy_2_fallback_rate": self.strategy_2_success_count
            / self.total_resolution_attempts,
            "strategy_3_success_rate": self.strategy_3_success_count
            / self.total_resolution_attempts,
            "strategy_4_success_rate": self.strategy_4_success_count
            / self.total_resolution_attempts,
            "failure_rate": self.failure_count / self.total_resolution_attempts,
            "average_resolution_time": sum(self.resolution_times)
            / len(self.resolution_times)
            if self.resolution_times
            else 0,
            "total_attempts": self.total_resolution_attempts,
        }


class HybridResolutionConfig:
    """Configuration for hybrid resolution rollout."""

    @staticmethod
    def is_hybrid_resolution_enabled() -> bool:
        """Check if hybrid resolution is enabled via environment variable."""
        return os.getenv("CURSUS_HYBRID_RESOLUTION_ENABLED", "true").lower() == "true"

    @staticmethod
    def get_hybrid_resolution_mode() -> str:
        """Get hybrid resolution mode: 'full', 'fallback_only', 'disabled'."""
        return os.getenv("CURSUS_HYBRID_RESOLUTION_MODE", "full")


# Global metrics instance
_hybrid_resolution_metrics = HybridResolutionMetrics()


def get_hybrid_resolution_metrics() -> Dict[str, Any]:
    """Get current hybrid resolution performance metrics."""
    return _hybrid_resolution_metrics.get_metrics()


class HybridPathResolver:
    """
    Hybrid path resolver that works across all deployment scenarios.

    This class implements the core hybrid resolution algorithm that uses
    Package Location Discovery first, then Working Directory Discovery as fallback.
    """

    def __init__(self):
        """Initialize the hybrid path resolver."""
        pass

    def resolve_path(
        self, project_root_folder: str, relative_path: str
    ) -> Optional[str]:
        """
        Hybrid path resolution: Package location first, then working directory discovery.

        This method implements the core hybrid resolution algorithm that works across
        all deployment scenarios:
        - Lambda/MODS bundled: Package location discovery
        - Development monorepo: Monorepo structure detection
        - Pip-installed separated: Working directory discovery fallback

        Args:
            project_root_folder: Root folder name for the user's project
            relative_path: Relative path from project root to target directory/file

        Returns:
            Resolved absolute path if found, None otherwise
        """
        if not relative_path:
            return None

        start_time = time.time()

        try:
            # Strategy 1: Package Location Discovery (works for all scenarios)
            resolved = self._package_location_discovery(
                project_root_folder, relative_path
            )
            if resolved:
                resolution_time = time.time() - start_time
                _hybrid_resolution_metrics.record_strategy_1_success(resolution_time)
                logger.info(
                    f"Hybrid resolution completed successfully via Package Location Discovery: {resolved}"
                )
                return resolved

            # Strategy 2: Working Directory Discovery (fallback for edge cases)
            resolved = self._working_directory_discovery(
                project_root_folder, relative_path
            )
            if resolved:
                resolution_time = time.time() - start_time
                _hybrid_resolution_metrics.record_strategy_2_success(resolution_time)
                logger.info(
                    f"Hybrid resolution completed successfully via Working Directory Discovery: {resolved}"
                )
                return resolved

            # Strategy 3: Generic Path Discovery (recursive search fallback)
            resolved = self._generic_path_discovery(project_root_folder, relative_path)
            if resolved:
                resolution_time = time.time() - start_time
                _hybrid_resolution_metrics.record_strategy_3_success(resolution_time)
                logger.info(
                    f"Hybrid resolution completed successfully via Generic Path Discovery: {resolved}"
                )
                return resolved

            # Strategy 4: Default Scripts Discovery (final fallback for cursus/steps/scripts)
            # This checks cursus/steps/scripts as a last resort when all other strategies fail
            resolved = self._default_scripts_discovery(relative_path)
            if resolved:
                resolution_time = time.time() - start_time
                _hybrid_resolution_metrics.record_strategy_4_success(resolution_time)
                logger.info(
                    f"Hybrid resolution completed successfully via Default Scripts Discovery: {resolved}"
                )
                return resolved

            # Resolution failed
            resolution_time = time.time() - start_time
            _hybrid_resolution_metrics.record_failure(resolution_time)
            logger.warning(
                f"Hybrid resolution failed - all strategies unsuccessful for project_root_folder='{project_root_folder}', relative_path='{relative_path}'"
            )
            return None

        except Exception as e:
            resolution_time = time.time() - start_time
            _hybrid_resolution_metrics.record_failure(resolution_time)
            logger.error(f"Hybrid resolution error: {e}")
            return None

    def _default_scripts_discovery(self, relative_path: str) -> Optional[str]:
        """
        Discover scripts using cursus package location to find default scripts directory.

        This strategy specifically looks for scripts in cursus/steps/scripts by:
        1. Starting from Path(__file__) which is in cursus/core/utils/
        2. Navigating up to cursus/ root
        3. Going to steps/scripts/
        4. Appending the relative_path (script filename or subdirectory path)

        This enables automatic discovery of scripts in the default location without
        requiring users to specify source_dir or project_root_folder.

        Args:
            relative_path: Relative path to script (can be just filename like "xgboost_training.py"
                          or subdirectory path like "calibration/standard_calibration_dictionary.json")

        Returns:
            Resolved absolute path if found, None otherwise
        """
        try:
            cursus_file = Path(__file__)  # cursus/core/utils/hybrid_path_resolution.py
            logger.debug(f"Default scripts discovery starting from: {cursus_file}")

            # Navigate from cursus/core/utils/ up to cursus/ then to steps/scripts/
            # cursus_file.parent = cursus/core/utils/
            # cursus_file.parent.parent = cursus/core/
            # cursus_file.parent.parent.parent = cursus/
            cursus_root = cursus_file.parent.parent.parent  # Go up 3 levels to cursus/
            scripts_dir = cursus_root / "steps" / "scripts"

            logger.debug(f"Checking default scripts directory: {scripts_dir}")

            if scripts_dir.exists() and scripts_dir.is_dir():
                target_path = scripts_dir / relative_path
                logger.debug(f"Checking default scripts path: {target_path}")

                if target_path.exists():
                    logger.info(f"Default scripts discovery succeeded: {target_path}")
                    return str(target_path)
                else:
                    logger.debug(f"Script not found at default location: {target_path}")
            else:
                logger.debug(f"Default scripts directory does not exist: {scripts_dir}")

            logger.debug(
                "Default scripts discovery failed - script not found in cursus/steps/scripts"
            )
            return None

        except Exception as e:
            logger.warning(f"Default scripts discovery failed with error: {e}")
            return None

    def _package_location_discovery(
        self, project_root_folder: Optional[str], relative_path: str
    ) -> Optional[str]:
        """
        Discover paths using cursus package location as reference.

        This strategy works across all deployment scenarios by using the cursus
        package location as a reference point to find project files.

        Args:
            project_root_folder: Root folder name for the user's project
            relative_path: Relative path from project root to target

        Returns:
            Resolved absolute path if found, None otherwise
        """
        try:
            cursus_file = Path(__file__)  # Current cursus module file
            logger.debug(
                f"Package location discovery starting from cursus file: {cursus_file}"
            )

            # Strategy 1A: Check for bundled deployment (Lambda/MODS)
            # Look for sibling directories to cursus
            potential_package_root = (
                cursus_file.parent.parent.parent.parent
            )  # Go up from cursus/core/utils/
            logger.debug(f"Checking potential package root: {potential_package_root}")

            # If project_root_folder is specified, use it directly
            if project_root_folder:
                direct_path = (
                    potential_package_root / project_root_folder / relative_path
                )
                logger.debug(f"Checking bundled deployment path: {direct_path}")
                if direct_path.exists():
                    logger.info(
                        f"Package location discovery succeeded (bundled): {direct_path}"
                    )
                    return str(direct_path)

            # Try direct resolution from package root (for backward compatibility)
            direct_path = potential_package_root / relative_path
            logger.debug(f"Checking direct package root path: {direct_path}")
            if direct_path.exists():
                logger.info(
                    f"Package location discovery succeeded (direct): {direct_path}"
                )
                return str(direct_path)

            # Strategy 1B: Check if we're in monorepo structure (src/cursus)
            if "src" in cursus_file.parts:
                src_index = cursus_file.parts.index("src")
                project_root = Path(*cursus_file.parts[:src_index])
                logger.debug(
                    f"Detected monorepo structure, project root: {project_root}"
                )

                if project_root.exists() and project_root.is_dir():
                    if project_root_folder:
                        target_path = project_root / project_root_folder / relative_path
                    else:
                        target_path = project_root / relative_path

                    logger.debug(f"Checking monorepo path: {target_path}")
                    if target_path.exists():
                        logger.info(
                            f"Package location discovery succeeded (monorepo): {target_path}"
                        )
                        return str(target_path)

            logger.debug("Package location discovery failed - no valid paths found")
            return None

        except Exception as e:
            logger.warning(f"Package location discovery failed with error: {e}")
            return None

    def _working_directory_discovery(
        self, project_root_folder: Optional[str], relative_path: str
    ) -> Optional[str]:
        """
        Discover paths using working directory traversal (fallback).

        This strategy searches upward from the current working directory to find
        project files, useful for pip-installed scenarios where package location
        discovery may not work.

        Args:
            project_root_folder: Root folder name for the user's project
            relative_path: Relative path from project root to target

        Returns:
            Resolved absolute path if found, None otherwise
        """
        try:
            current = Path.cwd()
            logger.debug(f"Working directory discovery starting from: {current}")

            # Search upward for project root
            search_depth = 0
            max_search_depth = 10  # Prevent infinite loops

            while current != current.parent and search_depth < max_search_depth:
                logger.debug(
                    f"Searching in directory: {current} (depth: {search_depth})"
                )

                # Strategy 2A: If project_root_folder is specified, check if we're inside it
                if project_root_folder:
                    # Check if current directory name matches project_root_folder
                    if current.name == project_root_folder:
                        target_path = current / relative_path
                        logger.debug(f"Checking project root match path: {target_path}")
                        if target_path.exists():
                            logger.info(
                                f"Working directory discovery succeeded (project root match): {target_path}"
                            )
                            return str(target_path)

                    # Check if project_root_folder exists as subdirectory of current
                    project_folder_path = current / project_root_folder
                    if project_folder_path.exists() and project_folder_path.is_dir():
                        target_path = project_folder_path / relative_path
                        logger.debug(
                            f"Checking project subdirectory path: {target_path}"
                        )
                        if target_path.exists():
                            logger.info(
                                f"Working directory discovery succeeded (project subdirectory): {target_path}"
                            )
                            return str(target_path)

                # Strategy 2B: Direct path resolution (for cases without project_root_folder)
                direct_path = current / relative_path
                logger.debug(f"Checking direct working directory path: {direct_path}")
                if direct_path.exists():
                    logger.info(
                        f"Working directory discovery succeeded (direct): {direct_path}"
                    )
                    return str(direct_path)

                current = current.parent
                search_depth += 1

            # Final fallback: try current working directory
            if project_root_folder:
                fallback_with_project = Path.cwd() / project_root_folder / relative_path
                logger.debug(
                    f"Checking final fallback with project: {fallback_with_project}"
                )
                if fallback_with_project.exists():
                    logger.info(
                        f"Working directory discovery succeeded (final fallback with project): {fallback_with_project}"
                    )
                    return str(fallback_with_project)

            fallback_path = Path.cwd() / relative_path
            logger.debug(f"Checking final fallback path: {fallback_path}")
            if fallback_path.exists():
                logger.info(
                    f"Working directory discovery succeeded (final fallback): {fallback_path}"
                )
                return str(fallback_path)

            logger.debug("Working directory discovery failed - no valid paths found")
            return None

        except Exception as e:
            logger.warning(f"Working directory discovery failed with error: {e}")
            return None

    def _generic_path_discovery(
        self, project_root_folder: Optional[str], relative_path: str
    ) -> Optional[str]:
        """
        Discover paths using generic recursive search (final fallback).

        This strategy uses multi-directional recursive search to find uniquely
        named project folders across different directory structures. It searches
        both upward and downward from multiple reference points.

        Args:
            project_root_folder: Root folder name for the user's project
            relative_path: Relative path from project root to target

        Returns:
            Resolved absolute path if found, None otherwise
        """
        if not project_root_folder:
            logger.debug(
                "Generic path discovery skipped: no project_root_folder specified"
            )
            return None

        try:
            logger.debug(
                f"Generic path discovery starting for project_root_folder='{project_root_folder}'"
            )

            # Use generic discovery to find the project folder
            project_folder = find_project_folder_generic(project_root_folder)

            if project_folder:
                # Construct target path by appending relative_path
                target_path = project_folder / relative_path

                logger.debug(f"Checking generic discovery target: {target_path}")

                if target_path.exists():
                    logger.info(f"Generic path discovery succeeded: {target_path}")
                    return str(target_path)
                else:
                    logger.debug(
                        f"Generic path discovery found project folder but target path doesn't exist: {target_path}"
                    )

            logger.debug("Generic path discovery failed - project folder not found")
            return None

        except Exception as e:
            logger.warning(f"Generic path discovery failed with error: {e}")
            return None


def resolve_hybrid_path(project_root_folder: str, relative_path: str) -> Optional[str]:
    """
    Convenience function for hybrid path resolution.

    Args:
        project_root_folder: Root folder name for the user's project
        relative_path: Relative path from project root to target

    Returns:
        Resolved absolute path if found, None otherwise
    """
    if not HybridResolutionConfig.is_hybrid_resolution_enabled():
        logger.debug("Hybrid resolution disabled, returning None")
        return None

    mode = HybridResolutionConfig.get_hybrid_resolution_mode()

    if mode == "disabled":
        logger.debug(f"Hybrid resolution mode '{mode}' disabled, returning None")
        return None

    resolver = HybridPathResolver()

    if mode == "full":
        # Full hybrid resolution
        return resolver.resolve_path(project_root_folder, relative_path)
    elif mode == "fallback_only":
        # Only use Working Directory Discovery
        return resolver._working_directory_discovery(project_root_folder, relative_path)
    else:
        # Unknown mode, default to full
        logger.warning(f"Unknown hybrid resolution mode '{mode}', defaulting to full")
        return resolver.resolve_path(project_root_folder, relative_path)

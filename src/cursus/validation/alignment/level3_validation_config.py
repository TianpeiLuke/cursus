"""
Level 3 Validation Configuration

Configuration class for Level 3 (Specification ↔ Dependencies) alignment validation
with configurable compatibility score thresholds.
"""

from typing import Dict, Any
from enum import Enum


class ValidationMode(Enum):
    """Validation modes with different threshold requirements."""

    STRICT = "strict"  # Current behavior (exact resolution required)
    RELAXED = "relaxed"  # Allow dependencies with reasonable compatibility
    PERMISSIVE = "permissive"  # Allow dependencies with minimal compatibility


class Level3ValidationConfig:
    """Configuration for Level 3 validation thresholds and behavior."""

    def __init__(self, mode: ValidationMode = ValidationMode.RELAXED):
        """
        Initialize validation configuration.

        Args:
            mode: Validation mode determining threshold strictness
        """
        self.mode = mode

        # Set thresholds based on mode
        if mode == ValidationMode.STRICT:
            self.PASS_THRESHOLD = 0.8  # ≥ 0.8: PASS
            self.WARNING_THRESHOLD = 0.7  # 0.7-0.79: WARNING
            self.ERROR_THRESHOLD = 0.5  # 0.5-0.69: ERROR
            # < 0.5: CRITICAL
        elif mode == ValidationMode.RELAXED:
            self.PASS_THRESHOLD = 0.6  # ≥ 0.6: PASS
            self.WARNING_THRESHOLD = 0.4  # 0.4-0.59: WARNING
            self.ERROR_THRESHOLD = 0.2  # 0.2-0.39: ERROR
            # < 0.2: CRITICAL
        elif mode == ValidationMode.PERMISSIVE:
            self.PASS_THRESHOLD = 0.3  # ≥ 0.3: PASS
            self.WARNING_THRESHOLD = 0.2  # 0.2-0.29: WARNING
            self.ERROR_THRESHOLD = 0.1  # 0.1-0.19: ERROR
            # < 0.1: CRITICAL
        else:
            # Default to relaxed mode
            self.PASS_THRESHOLD = 0.6
            self.WARNING_THRESHOLD = 0.4
            self.ERROR_THRESHOLD = 0.2

        # Resolution threshold for dependency resolver (should match or be lower than PASS_THRESHOLD)
        self.RESOLUTION_THRESHOLD = min(0.5, self.PASS_THRESHOLD)

        # Enable detailed scoring in reports
        self.INCLUDE_SCORE_BREAKDOWN = True
        self.INCLUDE_ALTERNATIVE_CANDIDATES = True
        self.MAX_ALTERNATIVE_CANDIDATES = 3

        # Logging configuration
        self.LOG_SUCCESSFUL_RESOLUTIONS = True
        self.LOG_FAILED_RESOLUTIONS = True
        self.LOG_SCORE_DETAILS = False  # Set to True for debugging

    def determine_severity_from_score(self, score: float, is_required: bool) -> str:
        """
        Determine issue severity based on compatibility score and requirement.

        Args:
            score: Compatibility score (0.0 to 1.0)
            is_required: Whether the dependency is required

        Returns:
            Severity level string
        """
        if score >= self.PASS_THRESHOLD:
            return "INFO"  # Should not happen in failed deps, but just in case
        elif score >= self.WARNING_THRESHOLD:
            return "WARNING" if not is_required else "ERROR"
        elif score >= self.ERROR_THRESHOLD:
            return "ERROR"
        else:
            return "CRITICAL"

    def should_pass_validation(self, score: float) -> bool:
        """
        Determine if a compatibility score should pass validation.

        Args:
            score: Compatibility score (0.0 to 1.0)

        Returns:
            True if score meets pass threshold
        """
        return score >= self.PASS_THRESHOLD

    def get_threshold_description(self) -> Dict[str, Any]:
        """
        Get human-readable description of current thresholds.

        Returns:
            Dictionary with threshold descriptions
        """
        return {
            "mode": self.mode.value,
            "thresholds": {
                "pass": f"≥ {self.PASS_THRESHOLD:.1f}",
                "warning": f"{self.WARNING_THRESHOLD:.1f} - {self.PASS_THRESHOLD - 0.01:.2f}",
                "error": f"{self.ERROR_THRESHOLD:.1f} - {self.WARNING_THRESHOLD - 0.01:.2f}",
                "critical": f"< {self.ERROR_THRESHOLD:.1f}",
            },
            "resolution_threshold": self.RESOLUTION_THRESHOLD,
            "description": self._get_mode_description(),
        }

    def _get_mode_description(self) -> str:
        """Get description of the current validation mode."""
        descriptions = {
            ValidationMode.STRICT: "Strict validation requiring high compatibility scores for all dependencies",
            ValidationMode.RELAXED: "Relaxed validation allowing reasonable compatibility matches",
            ValidationMode.PERMISSIVE: "Permissive validation for exploration and development phases",
        }
        return descriptions.get(self.mode, "Unknown validation mode")

    @classmethod
    def create_strict_config(cls) -> "Level3ValidationConfig":
        """Create a strict validation configuration."""
        return cls(ValidationMode.STRICT)

    @classmethod
    def create_relaxed_config(cls) -> "Level3ValidationConfig":
        """Create a relaxed validation configuration."""
        return cls(ValidationMode.RELAXED)

    @classmethod
    def create_permissive_config(cls) -> "Level3ValidationConfig":
        """Create a permissive validation configuration."""
        return cls(ValidationMode.PERMISSIVE)

    @classmethod
    def create_custom_config(
        cls, pass_threshold: float, warning_threshold: float, error_threshold: float
    ) -> "Level3ValidationConfig":
        """
        Create a custom validation configuration with specific thresholds.

        Args:
            pass_threshold: Minimum score for passing validation
            warning_threshold: Minimum score for warning level
            error_threshold: Minimum score for error level

        Returns:
            Custom Level3ValidationConfig instance
        """
        config = cls(ValidationMode.RELAXED)  # Start with relaxed as base
        config.PASS_THRESHOLD = pass_threshold
        config.WARNING_THRESHOLD = warning_threshold
        config.ERROR_THRESHOLD = error_threshold
        config.RESOLUTION_THRESHOLD = min(0.5, pass_threshold)
        return config


__all__ = ["Level3ValidationConfig", "ValidationMode"]

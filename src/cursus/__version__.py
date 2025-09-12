"""Version information for Cursus."""

import os
from pathlib import Path


def _get_version():
    """Get version from VERSION file with fallback logic."""
    # Try to find VERSION file in project root
    current_dir = Path(__file__).parent
    version_file_paths = [
        # Try project root (2 levels up from src/cursus/__version__.py)
        current_dir.parent.parent / "VERSION",
        # Try relative to current directory
        current_dir / "VERSION",
        # Try one level up
        current_dir.parent / "VERSION",
    ]

    for version_file in version_file_paths:
        try:
            if version_file.exists():
                return version_file.read_text().strip()
        except (OSError, IOError):
            continue

    # Fallback: try importlib.metadata (works for installed packages)
    try:
        from importlib.metadata import version

        return version("cursus")
    except ImportError:
        # Python < 3.8 fallback
        try:
            from importlib_metadata import version

            return version("cursus")
        except ImportError:
            pass
    except Exception:
        pass

    # Final fallback - return a default version
    return "1.2.3"


__version__ = _get_version()
__title__ = "cursus"
__description__ = "Automatic SageMaker Pipeline Generation from DAG Specifications"
__author__ = "Tianpei Xie"
__author_email__ = "unidoctor@gmail.com"
__license__ = "MIT"
__url__ = "https://github.com/TianpeiLuke/cursus"

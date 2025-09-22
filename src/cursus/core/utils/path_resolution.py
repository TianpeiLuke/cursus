"""Deployment-context-agnostic path resolution utilities."""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_package_relative_path(absolute_path: str) -> str:
    """
    Convert absolute path to package-relative path for deployment portability.
    
    This function solves the core problem by extracting package-relative paths
    from absolute paths, making configurations portable across deployment contexts.
    
    Args:
        absolute_path: Absolute path to convert
        
    Returns:
        Package-relative path or original path if conversion fails
    """
    if not absolute_path or not Path(absolute_path).is_absolute():
        return absolute_path  # Already relative or empty
    
    try:
        abs_path = Path(absolute_path)
        path_parts = abs_path.parts
        
        # Look for common package indicators in path
        package_indicators = ['cursus', 'buyer_abuse_mods_template', 'src']
        
        for indicator in package_indicators:
            if indicator in path_parts:
                indicator_index = path_parts.index(indicator)
                
                # If indicator is 'src', skip it and use next part as package
                if indicator == 'src' and indicator_index + 1 < len(path_parts):
                    package_index = indicator_index + 1
                    relative_parts = path_parts[package_index + 1:]
                else:
                    # Use parts after the package indicator
                    relative_parts = path_parts[indicator_index + 1:]
                
                if relative_parts:
                    return str(Path(*relative_parts))
        
        # Strategy 2: Fallback - return original path
        logger.debug(f"Could not convert to package-relative: {absolute_path}")
        return absolute_path
        
    except Exception as e:
        logger.debug(f"Path conversion failed for {absolute_path}: {e}")
        return absolute_path


def resolve_package_relative_path(relative_path: str) -> str:
    """
    Resolve package-relative path to absolute path in current deployment context.
    
    This function finds the package installation location and resolves
    relative paths to absolute paths that work in the current environment.
    It handles both child and sibling directory structures in package installations.
    
    Args:
        relative_path: Package-relative path to resolve
        
    Returns:
        Absolute path in current deployment context or original path if resolution fails
    """
    if not relative_path or Path(relative_path).is_absolute():
        return relative_path  # Already absolute or empty
    
    try:
        # Find package root using simple module inspection
        import cursus
        if hasattr(cursus, '__file__') and cursus.__file__:
            cursus_package_dir = Path(cursus.__file__).parent
            
            # Strategy 1: Try as child of cursus package (traditional structure)
            child_resolved_path = cursus_package_dir / relative_path
            if child_resolved_path.exists():
                return str(child_resolved_path.resolve())
            
            # Strategy 2: Try as sibling of cursus package (Lambda/deployment structure)
            # Go up one level from cursus package to find the common parent
            package_installation_root = cursus_package_dir.parent
            sibling_resolved_path = package_installation_root / relative_path
            if sibling_resolved_path.exists():
                return str(sibling_resolved_path.resolve())
            
            # Strategy 3: Return child path even if it doesn't exist (for consistency)
            # This maintains backward compatibility and allows caller to handle missing files
            return str(child_resolved_path.resolve())
        
        # Fallback: return original path
        logger.debug(f"Could not resolve package-relative path: {relative_path}")
        return relative_path
        
    except Exception as e:
        logger.debug(f"Path resolution failed for {relative_path}: {e}")
        return relative_path

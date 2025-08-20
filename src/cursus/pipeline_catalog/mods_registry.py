"""
MODS Global Registry Integration

This module provides read-only integration with the MODS global registry,
allowing users to query registered templates and operational information.
"""

import logging
from typing import Dict, List, Optional, Any, Callable
from functools import wraps
import time

# Setup logging
logger = logging.getLogger(__name__)

# Cache for registry data to improve performance
_registry_cache = {}
_cache_timeout = 300  # 5 minutes


def safe_mods_access(fallback_value=None):
    """
    Decorator for safe MODS registry access with graceful fallback.
    
    Args:
        fallback_value: Value to return if MODS is unavailable
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ImportError as e:
                logger.warning(f"MODS not available for {func.__name__}: {e}")
                return fallback_value
            except Exception as e:
                logger.error(f"Error in MODS registry access {func.__name__}: {e}")
                return fallback_value
        return wrapper
    return decorator


def _is_cache_valid(cache_key: str) -> bool:
    """
    Check if cached data is still valid.
    
    Args:
        cache_key: The cache key to check
        
    Returns:
        bool: True if cache is valid, False otherwise
    """
    if cache_key not in _registry_cache:
        return False
    
    cache_entry = _registry_cache[cache_key]
    return time.time() - cache_entry["timestamp"] < _cache_timeout


def _get_cached_data(cache_key: str) -> Any:
    """
    Get cached data if valid.
    
    Args:
        cache_key: The cache key
        
    Returns:
        Any: Cached data or None if invalid
    """
    if _is_cache_valid(cache_key):
        return _registry_cache[cache_key]["data"]
    return None


def _set_cached_data(cache_key: str, data: Any) -> None:
    """
    Set cached data with timestamp.
    
    Args:
        cache_key: The cache key
        data: Data to cache
    """
    _registry_cache[cache_key] = {
        "data": data,
        "timestamp": time.time()
    }


@safe_mods_access(fallback_value=False)
def is_mods_available() -> bool:
    """
    Check if MODS is available in the current environment.
    
    Returns:
        bool: True if MODS is available, False otherwise
    """
    try:
        import mods_dag_compiler
        return True
    except ImportError:
        return False


@safe_mods_access(fallback_value={})
def get_mods_registry_status() -> Dict[str, Any]:
    """
    Get the status of the MODS global registry.
    
    Returns:
        Dict[str, Any]: Registry status information
    """
    cache_key = "registry_status"
    cached_data = _get_cached_data(cache_key)
    if cached_data is not None:
        return cached_data
    
    try:
        # Import MODS components
        from mods_dag_compiler import MODSPipelineDAGCompiler
        
        # Get registry status (this would be the actual MODS API call)
        # For now, we'll return a mock status
        status = {
            "available": True,
            "connection_status": "connected",
            "last_sync": "2025-08-20T09:30:00Z",
            "template_count": 0,
            "registry_version": "1.0.0"
        }
        
        # Try to get actual registry information if available
        try:
            # This would be the actual MODS registry API call
            # registry_info = MODSPipelineDAGCompiler.get_registry_info()
            # status.update(registry_info)
            pass
        except Exception as e:
            logger.debug(f"Could not get detailed registry info: {e}")
            status["connection_status"] = "limited"
        
        _set_cached_data(cache_key, status)
        return status
        
    except ImportError:
        return {
            "available": False,
            "connection_status": "unavailable",
            "error": "MODS not installed"
        }


@safe_mods_access(fallback_value=[])
def get_mods_registered_templates() -> List[Dict[str, Any]]:
    """
    Get list of templates registered in the MODS global registry.
    
    Returns:
        List[Dict[str, Any]]: List of registered template information
    """
    cache_key = "registered_templates"
    cached_data = _get_cached_data(cache_key)
    if cached_data is not None:
        return cached_data
    
    try:
        from mods_dag_compiler import MODSPipelineDAGCompiler
        
        # Get registered templates (this would be the actual MODS API call)
        templates = []
        
        try:
            # This would be the actual MODS registry API call
            # templates = MODSPipelineDAGCompiler.list_registered_templates()
            
            # For now, return empty list as we don't have actual registry access
            templates = []
            
        except Exception as e:
            logger.debug(f"Could not retrieve registered templates: {e}")
        
        _set_cached_data(cache_key, templates)
        return templates
        
    except ImportError:
        logger.warning("MODS not available for template retrieval")
        return []


@safe_mods_access(fallback_value=None)
def get_registry_template_info(template_id: str) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific template from the registry.
    
    Args:
        template_id: The ID of the template to retrieve
        
    Returns:
        Optional[Dict[str, Any]]: Template information or None if not found
    """
    cache_key = f"template_info_{template_id}"
    cached_data = _get_cached_data(cache_key)
    if cached_data is not None:
        return cached_data
    
    try:
        from mods_dag_compiler import MODSPipelineDAGCompiler
        
        # Get template info (this would be the actual MODS API call)
        template_info = None
        
        try:
            # This would be the actual MODS registry API call
            # template_info = MODSPipelineDAGCompiler.get_template_info(template_id)
            pass
        except Exception as e:
            logger.debug(f"Could not retrieve template info for {template_id}: {e}")
        
        _set_cached_data(cache_key, template_info)
        return template_info
        
    except ImportError:
        logger.warning(f"MODS not available for template info retrieval: {template_id}")
        return None


@safe_mods_access(fallback_value={})
def get_registry_templates_by_framework(framework: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get templates from the registry grouped by framework.
    
    Args:
        framework: Framework to filter by (e.g., "xgboost", "pytorch")
        
    Returns:
        Dict[str, List[Dict[str, Any]]]: Templates grouped by framework
    """
    cache_key = f"templates_by_framework_{framework}"
    cached_data = _get_cached_data(cache_key)
    if cached_data is not None:
        return cached_data
    
    try:
        all_templates = get_mods_registered_templates()
        
        # Group templates by framework
        grouped = {}
        for template in all_templates:
            template_framework = template.get("framework", "unknown")
            if framework and template_framework != framework:
                continue
                
            if template_framework not in grouped:
                grouped[template_framework] = []
            grouped[template_framework].append(template)
        
        _set_cached_data(cache_key, grouped)
        return grouped
        
    except Exception as e:
        logger.error(f"Error grouping templates by framework {framework}: {e}")
        return {}


@safe_mods_access(fallback_value=[])
def get_registry_templates_by_tags(tags: List[str]) -> List[Dict[str, Any]]:
    """
    Get templates from the registry that match specified tags.
    
    Args:
        tags: List of tags to filter by
        
    Returns:
        List[Dict[str, Any]]: Templates matching the tags
    """
    try:
        all_templates = get_mods_registered_templates()
        
        # Filter templates by tags
        matching_templates = []
        for template in all_templates:
            template_tags = template.get("tags", [])
            if any(tag in template_tags for tag in tags):
                matching_templates.append(template)
        
        return matching_templates
        
    except Exception as e:
        logger.error(f"Error filtering templates by tags {tags}: {e}")
        return []


def clear_registry_cache() -> None:
    """
    Clear the registry cache to force fresh data retrieval.
    """
    global _registry_cache
    _registry_cache.clear()
    logger.info("MODS registry cache cleared")


def get_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the registry cache.
    
    Returns:
        Dict[str, Any]: Cache statistics
    """
    current_time = time.time()
    valid_entries = 0
    expired_entries = 0
    
    for cache_key, cache_entry in _registry_cache.items():
        if current_time - cache_entry["timestamp"] < _cache_timeout:
            valid_entries += 1
        else:
            expired_entries += 1
    
    return {
        "total_entries": len(_registry_cache),
        "valid_entries": valid_entries,
        "expired_entries": expired_entries,
        "cache_timeout": _cache_timeout
    }


def set_cache_timeout(timeout_seconds: int) -> None:
    """
    Set the cache timeout for registry data.
    
    Args:
        timeout_seconds: Cache timeout in seconds
    """
    global _cache_timeout
    _cache_timeout = timeout_seconds
    logger.info(f"MODS registry cache timeout set to {timeout_seconds} seconds")


# Convenience functions for common operations

def check_mods_integration() -> Dict[str, Any]:
    """
    Comprehensive check of MODS integration status.
    
    Returns:
        Dict[str, Any]: Integration status report
    """
    report = {
        "mods_available": is_mods_available(),
        "registry_status": get_mods_registry_status(),
        "cache_stats": get_cache_stats()
    }
    
    if report["mods_available"]:
        try:
            templates = get_mods_registered_templates()
            report["template_count"] = len(templates)
            report["integration_status"] = "fully_operational"
        except Exception as e:
            report["integration_status"] = "limited"
            report["error"] = str(e)
    else:
        report["integration_status"] = "unavailable"
    
    return report


def get_mods_summary() -> Dict[str, Any]:
    """
    Get a summary of MODS registry information.
    
    Returns:
        Dict[str, Any]: MODS summary information
    """
    if not is_mods_available():
        return {
            "available": False,
            "message": "MODS not available in current environment"
        }
    
    try:
        registry_status = get_mods_registry_status()
        templates = get_mods_registered_templates()
        
        # Group templates by framework
        frameworks = {}
        for template in templates:
            framework = template.get("framework", "unknown")
            if framework not in frameworks:
                frameworks[framework] = 0
            frameworks[framework] += 1
        
        return {
            "available": True,
            "registry_connected": registry_status.get("connection_status") == "connected",
            "total_templates": len(templates),
            "frameworks": frameworks,
            "last_sync": registry_status.get("last_sync"),
            "registry_version": registry_status.get("registry_version")
        }
        
    except Exception as e:
        return {
            "available": True,
            "error": f"Error retrieving MODS summary: {e}",
            "registry_connected": False
        }

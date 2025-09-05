"""
Enhanced step names registry with hybrid backend support.
Maintains 100% backward compatibility while adding workspace awareness.

This module provides a drop-in replacement for the original step_names.py that:
- Uses the hybrid registry backend transparently
- Maintains all existing functions and variables
- Adds workspace context management capabilities
- Provides seamless workspace-aware step resolution
"""

import os
import logging
from typing import Dict, List, Optional, ContextManager
from contextlib import contextmanager

# Set up logging
logger = logging.getLogger(__name__)

# Global workspace context management
_current_workspace_context: Optional[str] = None

def set_workspace_context(workspace_id: str) -> None:
    """
    Set current workspace context for registry resolution.
    
    Args:
        workspace_id: Workspace identifier to set as current context
    """
    global _current_workspace_context
    _current_workspace_context = workspace_id
    logger.debug(f"Set workspace context to: {workspace_id}")

def get_workspace_context() -> Optional[str]:
    """
    Get current workspace context.
    
    Returns:
        Current workspace identifier or None if no context set
    """
    # Check explicit context first
    if _current_workspace_context:
        return _current_workspace_context
    
    # Check environment variable
    env_context = os.environ.get('CURSUS_WORKSPACE_ID')
    if env_context:
        return env_context
    
    return None

def clear_workspace_context() -> None:
    """Clear current workspace context."""
    global _current_workspace_context
    _current_workspace_context = None
    logger.debug("Cleared workspace context")

@contextmanager
def workspace_context(workspace_id: str) -> ContextManager[None]:
    """
    Context manager for temporary workspace context.
    
    Args:
        workspace_id: Workspace identifier for temporary context
        
    Usage:
        with workspace_context("developer_1"):
            step_names = get_step_names()  # Uses developer_1 context
    """
    old_context = get_workspace_context()
    try:
        set_workspace_context(workspace_id)
        yield
    finally:
        if old_context:
            set_workspace_context(old_context)
        else:
            clear_workspace_context()

# Global registry manager instance
_global_registry_manager = None

def _get_registry_manager():
    """Get or create global registry manager instance."""
    global _global_registry_manager
    if _global_registry_manager is None:
        try:
            from .hybrid.manager import UnifiedRegistryManager
            _global_registry_manager = UnifiedRegistryManager()
            logger.debug("Initialized hybrid registry manager")
        except Exception as e:
            logger.warning(f"Failed to initialize hybrid registry manager: {e}")
            # Fallback to original implementation
            _global_registry_manager = _create_fallback_manager()
    return _global_registry_manager

def _create_fallback_manager():
    """Create fallback manager using original step_names data."""
    logger.info("Using fallback registry manager with original step_names")
    
    class FallbackManager:
        def __init__(self):
            # Import original step names
            from .step_names_original import STEP_NAMES as ORIGINAL_STEP_NAMES
            self._step_names = ORIGINAL_STEP_NAMES
        
        def create_legacy_step_names_dict(self, workspace_id: str = None) -> Dict[str, Dict[str, str]]:
            return self._step_names.copy()
        
        def get_step_definition(self, step_name: str, workspace_id: str = None):
            return self._step_names.get(step_name)
        
        def has_step(self, step_name: str, workspace_id: str = None) -> bool:
            return step_name in self._step_names
        
        def list_steps(self, workspace_id: str = None) -> List[str]:
            return list(self._step_names.keys())
    
    return FallbackManager()

# Core registry data structures with workspace awareness
def get_step_names(workspace_id: str = None) -> Dict[str, Dict[str, str]]:
    """
    Get STEP_NAMES dictionary with workspace context.
    
    Args:
        workspace_id: Optional workspace context override
        
    Returns:
        Step names dictionary in original format
    """
    effective_workspace = workspace_id or get_workspace_context()
    manager = _get_registry_manager()
    return manager.create_legacy_step_names_dict(effective_workspace)

# Dynamic registry variables that update with workspace context
@property
def STEP_NAMES() -> Dict[str, Dict[str, str]]:
    """Dynamic STEP_NAMES that respects workspace context."""
    return get_step_names()

# Generate derived registries dynamically
def get_config_step_registry(workspace_id: str = None) -> Dict[str, str]:
    """Get CONFIG_STEP_REGISTRY with workspace context."""
    step_names = get_step_names(workspace_id)
    return {
        info["config_class"]: step_name 
        for step_name, info in step_names.items()
        if "config_class" in info
    }

def get_builder_step_names(workspace_id: str = None) -> Dict[str, str]:
    """Get BUILDER_STEP_NAMES with workspace context."""
    step_names = get_step_names(workspace_id)
    return {
        step_name: info["builder_step_name"]
        for step_name, info in step_names.items()
        if "builder_step_name" in info
    }

def get_spec_step_types(workspace_id: str = None) -> Dict[str, str]:
    """Get SPEC_STEP_TYPES with workspace context."""
    step_names = get_step_names(workspace_id)
    return {
        step_name: info["spec_type"]
        for step_name, info in step_names.items()
        if "spec_type" in info
    }

# Backward compatibility: Create module-level variables
# These will be dynamically updated based on workspace context
STEP_NAMES = get_step_names()
CONFIG_STEP_REGISTRY = get_config_step_registry()
BUILDER_STEP_NAMES = get_builder_step_names()
SPEC_STEP_TYPES = get_spec_step_types()

# Helper functions with workspace awareness
def get_config_class_name(step_name: str, workspace_id: str = None) -> str:
    """Get config class name with workspace context."""
    step_names = get_step_names(workspace_id)
    if step_name not in step_names:
        available_steps = sorted(step_names.keys())
        raise ValueError(f"Unknown step name: {step_name}. Available steps: {available_steps}")
    return step_names[step_name]["config_class"]

def get_builder_step_name(step_name: str, workspace_id: str = None) -> str:
    """Get builder step class name with workspace context."""
    step_names = get_step_names(workspace_id)
    if step_name not in step_names:
        available_steps = sorted(step_names.keys())
        raise ValueError(f"Unknown step name: {step_name}. Available steps: {available_steps}")
    return step_names[step_name]["builder_step_name"]

def get_spec_step_type(step_name: str, workspace_id: str = None) -> str:
    """Get step_type value for StepSpecification with workspace context."""
    step_names = get_step_names(workspace_id)
    if step_name not in step_names:
        available_steps = sorted(step_names.keys())
        raise ValueError(f"Unknown step name: {step_name}. Available steps: {available_steps}")
    return step_names[step_name]["spec_type"]

def get_spec_step_type_with_job_type(step_name: str, job_type: str = None, workspace_id: str = None) -> str:
    """Get step_type with optional job_type suffix, workspace-aware."""
    base_type = get_spec_step_type(step_name, workspace_id)
    if job_type:
        return f"{base_type}_{job_type.capitalize()}"
    return base_type

def get_step_name_from_spec_type(spec_type: str, workspace_id: str = None) -> str:
    """Get canonical step name from spec_type with workspace context."""
    base_spec_type = spec_type.split('_')[0] if '_' in spec_type else spec_type
    step_names = get_step_names(workspace_id)
    reverse_mapping = {info["spec_type"]: step_name for step_name, info in step_names.items()}
    return reverse_mapping.get(base_spec_type, spec_type)

def get_all_step_names(workspace_id: str = None) -> List[str]:
    """Get all canonical step names with workspace context."""
    step_names = get_step_names(workspace_id)
    return list(step_names.keys())

def validate_step_name(step_name: str, workspace_id: str = None) -> bool:
    """Validate step name exists with workspace context."""
    step_names = get_step_names(workspace_id)
    return step_name in step_names

def validate_spec_type(spec_type: str, workspace_id: str = None) -> bool:
    """Validate spec_type exists with workspace context."""
    base_spec_type = spec_type.split('_')[0] if '_' in spec_type else spec_type
    step_names = get_step_names(workspace_id)
    return base_spec_type in [info["spec_type"] for info in step_names.values()]

def get_step_description(step_name: str, workspace_id: str = None) -> str:
    """Get step description with workspace context."""
    step_names = get_step_names(workspace_id)
    if step_name not in step_names:
        available_steps = sorted(step_names.keys())
        raise ValueError(f"Unknown step name: {step_name}. Available steps: {available_steps}")
    return step_names[step_name]["description"]

def list_all_step_info(workspace_id: str = None) -> Dict[str, Dict[str, str]]:
    """Get complete step information with workspace context."""
    return get_step_names(workspace_id)

# SageMaker Step Type Classification Functions with workspace awareness
def get_sagemaker_step_type(step_name: str, workspace_id: str = None) -> str:
    """Get SageMaker step type with workspace context."""
    step_names = get_step_names(workspace_id)
    if step_name not in step_names:
        available_steps = sorted(step_names.keys())
        raise ValueError(f"Unknown step name: {step_name}. Available steps: {available_steps}")
    return step_names[step_name]["sagemaker_step_type"]

def get_steps_by_sagemaker_type(sagemaker_type: str, workspace_id: str = None) -> List[str]:
    """Get steps by SageMaker type with workspace context."""
    step_names = get_step_names(workspace_id)
    return [
        step_name for step_name, info in step_names.items()
        if info["sagemaker_step_type"] == sagemaker_type
    ]

def get_all_sagemaker_step_types(workspace_id: str = None) -> List[str]:
    """Get all SageMaker step types with workspace context."""
    step_names = get_step_names(workspace_id)
    return list(set(info["sagemaker_step_type"] for info in step_names.values()))

def validate_sagemaker_step_type(sagemaker_type: str) -> bool:
    """Validate SageMaker step type."""
    valid_types = {"Processing", "Training", "Transform", "CreateModel", "RegisterModel", "Base", "Utility", "Lambda", "CradleDataLoading", "MimsModelRegistrationProcessing"}
    return sagemaker_type in valid_types

def get_sagemaker_step_type_mapping(workspace_id: str = None) -> Dict[str, List[str]]:
    """Get SageMaker step type mapping with workspace context."""
    step_names = get_step_names(workspace_id)
    mapping = {}
    for step_name, info in step_names.items():
        sagemaker_type = info["sagemaker_step_type"]
        if sagemaker_type not in mapping:
            mapping[sagemaker_type] = []
        mapping[sagemaker_type].append(step_name)
    return mapping

def get_canonical_name_from_file_name(file_name: str, workspace_id: str = None) -> str:
    """
    Enhanced file name resolution with workspace context awareness.
    
    Args:
        file_name: File-based name (e.g., "model_evaluation_xgb", "dummy_training")
        workspace_id: Optional workspace context for resolution
        
    Returns:
        Canonical step name (e.g., "XGBoostModelEval", "DummyTraining")
        
    Raises:
        ValueError: If file name cannot be mapped to a canonical name
    """
    if not file_name:
        raise ValueError("File name cannot be empty")
    
    # Get workspace-aware step names
    step_names = get_step_names(workspace_id)
    
    parts = file_name.split('_')
    job_type_suffixes = ['training', 'validation', 'testing', 'calibration']
    
    # Strategy 1: Try full name as PascalCase
    full_pascal = ''.join(word.capitalize() for word in parts)
    if full_pascal in step_names:
        return full_pascal
    
    # Strategy 2: Try without last part if it's a job type suffix
    if len(parts) > 1 and parts[-1] in job_type_suffixes:
        base_parts = parts[:-1]
        base_pascal = ''.join(word.capitalize() for word in base_parts)
        if base_pascal in step_names:
            return base_pascal
    
    # Strategy 3: Handle special abbreviations and patterns
    abbreviation_map = {
        'xgb': 'XGBoost',
        'xgboost': 'XGBoost',
        'pytorch': 'PyTorch',
        'mims': '',
        'tabular': 'Tabular',
        'preprocess': 'Preprocessing'
    }
    
    # Apply abbreviation expansion
    expanded_parts = []
    for part in parts:
        if part in abbreviation_map:
            expansion = abbreviation_map[part]
            if expansion:
                expanded_parts.append(expansion)
        else:
            expanded_parts.append(part.capitalize())
    
    # Try expanded version
    if expanded_parts:
        expanded_pascal = ''.join(expanded_parts)
        if expanded_pascal in step_names:
            return expanded_pascal
        
        # Try expanded version without job type suffix
        if len(expanded_parts) > 1 and parts[-1] in job_type_suffixes:
            expanded_base = ''.join(expanded_parts[:-1])
            if expanded_base in step_names:
                return expanded_base
    
    # Strategy 4: Handle compound names (like "model_evaluation_xgb")
    if len(parts) >= 3:
        combinations_to_try = [
            (parts[-1], parts[0], parts[1]),  # xgb, model, evaluation → XGBoost, Model, Eval
            (parts[0], parts[1], parts[-1]),  # model, evaluation, xgb
        ]
        
        for combo in combinations_to_try:
            expanded_combo = []
            for part in combo:
                if part in abbreviation_map:
                    expansion = abbreviation_map[part]
                    if expansion:
                        expanded_combo.append(expansion)
                else:
                    if part == 'evaluation':
                        expanded_combo.append('Eval')
                    else:
                        expanded_combo.append(part.capitalize())
            
            combo_pascal = ''.join(expanded_combo)
            if combo_pascal in step_names:
                return combo_pascal
    
    # Strategy 5: Fuzzy matching against registry entries
    best_match = None
    best_score = 0.0
    
    for canonical_name in step_names.keys():
        score = _calculate_name_similarity(file_name, canonical_name)
        if score > best_score and score >= 0.8:
            best_score = score
            best_match = canonical_name
    
    if best_match:
        return best_match
    
    # Enhanced error message with workspace context
    tried_variations = [
        full_pascal,
        ''.join(word.capitalize() for word in parts[:-1]) if len(parts) > 1 and parts[-1] in job_type_suffixes else None,
        ''.join(expanded_parts) if expanded_parts else None
    ]
    tried_variations = [v for v in tried_variations if v]
    
    workspace_context = get_workspace_context()
    context_info = f" (workspace: {workspace_context})" if workspace_context else " (core registry)"
    
    raise ValueError(
        f"Cannot map file name '{file_name}' to canonical name{context_info}. "
        f"Tried variations: {tried_variations}. "
        f"Available canonical names: {sorted(step_names.keys())}"
    )

def _calculate_name_similarity(file_name: str, canonical_name: str) -> float:
    """Calculate similarity score between file name and canonical name."""
    file_lower = file_name.lower().replace('_', '')
    canonical_lower = canonical_name.lower()
    
    if file_lower == canonical_lower:
        return 1.0
    
    if file_lower in canonical_lower:
        return 0.9
    
    file_parts = file_name.lower().split('_')
    matches = sum(1 for part in file_parts if part in canonical_lower)
    
    if matches == len(file_parts):
        return 0.85
    elif matches >= len(file_parts) * 0.8:
        return 0.8
    else:
        return matches / len(file_parts) * 0.7

def validate_file_name(file_name: str, workspace_id: str = None) -> bool:
    """Validate file name can be mapped with workspace context."""
    try:
        get_canonical_name_from_file_name(file_name, workspace_id)
        return True
    except ValueError:
        return False

# Workspace management functions
def list_available_workspaces() -> List[str]:
    """List all available workspace contexts."""
    try:
        manager = _get_registry_manager()
        if hasattr(manager, 'get_registry_status'):
            status = manager.get_registry_status()
            return [ws_id for ws_id in status.keys() if ws_id != 'core']
        return []
    except Exception as e:
        logger.warning(f"Failed to list workspaces: {e}")
        return []

def get_workspace_step_count(workspace_id: str) -> int:
    """Get number of steps available in a workspace."""
    try:
        manager = _get_registry_manager()
        if hasattr(manager, 'get_step_count'):
            return manager.get_step_count(workspace_id)
        return len(get_step_names(workspace_id))
    except Exception as e:
        logger.warning(f"Failed to get step count for workspace {workspace_id}: {e}")
        return 0

def has_workspace_conflicts() -> bool:
    """Check if there are any step name conflicts between workspaces."""
    try:
        manager = _get_registry_manager()
        if hasattr(manager, 'get_step_conflicts'):
            conflicts = manager.get_step_conflicts()
            return len(conflicts) > 0
        return False
    except Exception as e:
        logger.warning(f"Failed to check workspace conflicts: {e}")
        return False

# Update module-level variables when workspace context changes
def _refresh_module_variables():
    """Refresh module-level variables with current workspace context."""
    global STEP_NAMES, CONFIG_STEP_REGISTRY, BUILDER_STEP_NAMES, SPEC_STEP_TYPES
    current_workspace = get_workspace_context()
    STEP_NAMES = get_step_names(current_workspace)
    CONFIG_STEP_REGISTRY = get_config_step_registry(current_workspace)
    BUILDER_STEP_NAMES = get_builder_step_names(current_workspace)
    SPEC_STEP_TYPES = get_spec_step_types(current_workspace)

# Auto-refresh variables when workspace context is set
def _set_workspace_context_with_refresh(workspace_id: str) -> None:
    """Set workspace context and refresh module variables."""
    global _current_workspace_context
    _current_workspace_context = workspace_id
    _refresh_module_variables()

# Override the original set_workspace_context to include refresh
original_set_workspace_context = set_workspace_context
def set_workspace_context(workspace_id: str) -> None:
    """Set workspace context and refresh module variables."""
    original_set_workspace_context(workspace_id)
    _refresh_module_variables()

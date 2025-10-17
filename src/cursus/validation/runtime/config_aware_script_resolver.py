"""
Config-Aware Script Path Resolver for Runtime Validation.

This module implements the unified script path resolution system that replaces
unreliable discovery chains with a single, reliable resolution component using
config instances + hybrid path resolution.

The resolver eliminates:
- Name conversion logic (80+ lines)
- Multi-tier discovery chains (150+ lines) 
- Fuzzy matching (50+ lines)
- Placeholder script creation (30+ lines)
- Complex workspace discovery (100+ lines)

And replaces them with a single reliable method using the same proven approach
as step builders: config instances + hybrid path resolution.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConfigAwareScriptPathResolver:
    """
    Unified script path resolver using config instances + hybrid resolution.
    
    This class replaces ALL unreliable discovery methods with the proven approach
    used by step builders: extract entry points from config instances and use
    hybrid path resolution for deployment-agnostic file location.
    
    Features:
    - Config instance-based entry point extraction (eliminates phantom scripts)
    - Hybrid path resolution integration (deployment-agnostic)
    - No name conversion needed (config contains exact entry points)
    - No fuzzy matching needed (config validation ensures accuracy)
    - No placeholder creation (config validation prevents missing scripts)
    - Single method replaces entire discovery chain
    
    Eliminated Methods (from runtime_spec_builder.py):
    - _canonical_to_script_name() - No name conversion needed
    - _find_script_file() - Single resolve_script_path() method
    - _find_in_workspace() - Hybrid resolution handles this
    - _find_fuzzy_match() - Config validation eliminates need
    - _create_placeholder_script() - Config validation prevents missing scripts
    """
    
    def __init__(self):
        """Initialize unified resolver with minimal dependencies."""
        self.logger = logging.getLogger(__name__)
    
    def resolve_script_path(self, config_instance) -> Optional[str]:
        """
        Resolve script path using config instance + hybrid resolution.
        
        This is the ONLY method needed - replaces entire discovery chain.
        Uses the same proven approach as step builders:
        1. Extract entry point from config instance (no name conversion)
        2. Use hybrid resolution for deployment-agnostic file location
        3. Return absolute path or None (no fake scripts)
        
        Args:
            config_instance: Config instance containing entry point and source directory
            
        Returns:
            Absolute path to script file or None if no script for this config
            
        Example:
            >>> resolver = ConfigAwareScriptPathResolver()
            >>> config = ProcessingStepConfigBase(
            ...     processing_source_dir="scripts",
            ...     processing_entry_point="tabular_preprocessing.py"
            ... )
            >>> script_path = resolver.resolve_script_path(config)
            >>> # Returns: "/absolute/path/to/scripts/tabular_preprocessing.py"
        """
        # Step 1: Extract entry point from config (eliminates phantom scripts)
        entry_point = self._extract_entry_point_from_config(config_instance)
        if not entry_point:
            self.logger.debug(f"No entry point found in config {type(config_instance).__name__}")
            return None  # No script for this config
        
        # Step 2: Use config's built-in hybrid resolution (preferred)
        if hasattr(config_instance, 'get_resolved_script_path'):
            resolved = config_instance.get_resolved_script_path()
            if resolved and Path(resolved).exists():
                self.logger.info(f"✅ Script resolved via config method: {resolved}")
                return resolved
        
        # Step 3: Manual hybrid resolution using config's source directory
        source_dir = self._get_effective_source_dir(config_instance)
        if source_dir:
            # Use hybrid_path_resolution for deployment-agnostic resolution
            try:
                from ...core.utils.hybrid_path_resolution import resolve_hybrid_path
                
                # Construct relative path from project root
                relative_path = f"{source_dir}/{entry_point}"
                resolved = resolve_hybrid_path(
                    project_root_folder=None,  # Let hybrid resolution find project root
                    relative_path=relative_path
                )
                if resolved and Path(resolved).exists():
                    self.logger.info(f"✅ Script resolved via hybrid resolution: {resolved}")
                    return resolved
            except ImportError:
                self.logger.warning("Hybrid path resolution not available, falling back to direct path construction")
                # Fallback: direct path construction
                direct_path = Path(source_dir) / entry_point
                if direct_path.exists():
                    self.logger.info(f"✅ Script resolved via direct path: {direct_path}")
                    return str(direct_path)
        
        # Step 4: No fallbacks needed - config validation ensures script exists
        self.logger.warning(f"Script not found for entry point: {entry_point}")
        return None
    
    def _extract_entry_point_from_config(self, config_instance) -> Optional[str]:
        """
        Extract entry point from config instance (no name conversion needed).
        
        Checks common entry point field names used across different config types.
        This eliminates the need for complex name conversion logic since config
        instances already contain the exact entry point filenames.
        
        Args:
            config_instance: Config instance to extract entry point from
            
        Returns:
            Entry point filename or None if no entry point found
        """
        # Check common entry point field names in priority order
        entry_point_fields = [
            'processing_entry_point',    # ProcessingStepConfigBase
            'training_entry_point',      # TrainingStepConfigBase  
            'inference_entry_point',     # InferenceStepConfigBase
            'entry_point'                # Generic entry point
        ]
        
        for field in entry_point_fields:
            if hasattr(config_instance, field):
                entry_point = getattr(config_instance, field)
                if entry_point:
                    self.logger.debug(f"Found entry point '{entry_point}' in field '{field}'")
                    return entry_point
        
        self.logger.debug(f"No entry point found in config {type(config_instance).__name__}")
        return None
    
    def _get_effective_source_dir(self, config_instance) -> Optional[str]:
        """
        Get effective source directory from config instance.
        
        Checks common source directory field names and uses hybrid resolution
        when available. This eliminates the need for complex workspace discovery
        since config instances already contain source directory information.
        
        Args:
            config_instance: Config instance to extract source directory from
            
        Returns:
            Source directory path or None if no source directory found
        """
        # Check common source directory field names (in priority order)
        source_dir_fields = [
            'resolved_processing_source_dir',  # Hybrid-resolved directory (preferred)
            'effective_source_dir',            # Effective directory property
            'processing_source_dir',           # Processing-specific source directory
            'source_dir'                       # Generic source directory
        ]
        
        for field in source_dir_fields:
            if hasattr(config_instance, field):
                source_dir = getattr(config_instance, field)
                if source_dir:
                    self.logger.debug(f"Found source directory '{source_dir}' in field '{field}'")
                    return source_dir
        
        self.logger.debug(f"No source directory found in config {type(config_instance).__name__}")
        return None
    
    def validate_config_for_script_resolution(self, config_instance) -> Dict[str, Any]:
        """
        Validate config instance for script resolution capability.
        
        Returns comprehensive validation report with entry point and source
        directory status. This helps with debugging and provides clear feedback
        about why script resolution may fail.
        
        Args:
            config_instance: Config instance to validate
            
        Returns:
            Validation report dictionary with detailed status information
            
        Example:
            >>> resolver = ConfigAwareScriptPathResolver()
            >>> validation = resolver.validate_config_for_script_resolution(config)
            >>> print(validation)
            {
                'config_type': 'ProcessingStepConfigBase',
                'has_entry_point': True,
                'entry_point': 'tabular_preprocessing.py',
                'has_source_dir': True,
                'source_dir': 'scripts',
                'can_resolve_script': True,
                'resolution_method': 'config_method'
            }
        """
        entry_point = self._extract_entry_point_from_config(config_instance)
        source_dir = self._get_effective_source_dir(config_instance)
        
        return {
            'config_type': type(config_instance).__name__,
            'has_entry_point': entry_point is not None,
            'entry_point': entry_point,
            'has_source_dir': source_dir is not None,
            'source_dir': source_dir,
            'can_resolve_script': entry_point is not None and source_dir is not None,
            'resolution_method': 'config_method' if hasattr(config_instance, 'get_resolved_script_path') else 'hybrid_resolution'
        }
    
    def get_resolution_report(self, config_instances: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive resolution report for multiple config instances.
        
        Analyzes multiple config instances and provides summary statistics
        about script resolution capabilities. Useful for debugging and
        understanding which configs have scripts vs. data-only transformations.
        
        Args:
            config_instances: Dictionary mapping names to config instances
            
        Returns:
            Resolution report with summary statistics and detailed breakdown
            
        Example:
            >>> resolver = ConfigAwareScriptPathResolver()
            >>> report = resolver.get_resolution_report(loaded_configs)
            >>> print(f"Scripts found: {report['scripts_found']}")
            >>> print(f"Phantom scripts eliminated: {report['phantom_scripts_eliminated']}")
        """
        total_configs = len(config_instances)
        scripts_found = 0
        phantom_scripts_eliminated = 0
        resolution_details = {}
        
        for name, config_instance in config_instances.items():
            validation = self.validate_config_for_script_resolution(config_instance)
            resolution_details[name] = validation
            
            if validation['can_resolve_script']:
                # Try to actually resolve the script
                script_path = self.resolve_script_path(config_instance)
                if script_path:
                    scripts_found += 1
                    validation['script_path'] = script_path
                    validation['resolution_status'] = 'success'
                else:
                    validation['resolution_status'] = 'failed'
            else:
                # Config without entry point - phantom script eliminated
                phantom_scripts_eliminated += 1
                validation['resolution_status'] = 'no_script'
        
        return {
            'total_configs': total_configs,
            'scripts_found': scripts_found,
            'phantom_scripts_eliminated': phantom_scripts_eliminated,
            'resolution_success_rate': scripts_found / total_configs if total_configs > 0 else 0.0,
            'phantom_elimination_rate': phantom_scripts_eliminated / total_configs if total_configs > 0 else 0.0,
            'resolution_details': resolution_details,
            'summary': {
                'reliable_discovery': True,
                'deployment_agnostic': True,
                'no_fuzzy_matching': True,
                'no_placeholder_scripts': True,
                'config_based_validation': True
            }
        }

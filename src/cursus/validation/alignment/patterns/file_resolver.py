"""
Hybrid File Resolution Engine

Provides advanced file resolution strategies with production registry integration,
fuzzy matching, and multiple fallback mechanisms.
"""

import re
from typing import Optional, Dict, Any
from pathlib import Path

from ..alignment_utils import FlexibleFileResolver


class HybridFileResolver:
    """
    Advanced file resolver with multiple resolution strategies.
    
    Combines:
    - Production registry mapping
    - Standard naming conventions
    - FlexibleFileResolver patterns
    - Fuzzy matching fallbacks
    """
    
    def __init__(self, base_directories: Dict[str, str]):
        """
        Initialize the hybrid file resolver.
        
        Args:
            base_directories: Dictionary mapping directory types to paths
        """
        self.base_directories = base_directories
        self.flexible_resolver = FlexibleFileResolver(base_directories)
        
        # Initialize directory paths
        self.builders_dir = Path(base_directories.get('builders', ''))
        self.configs_dir = Path(base_directories.get('configs', ''))
    
    def find_builder_file(self, builder_name: str) -> Optional[str]:
        """
        Hybrid builder file resolution with multiple fallback strategies.
        
        Priority:
        1. Standard pattern: builder_{builder_name}_step.py
        2. FlexibleFileResolver patterns (includes fuzzy matching)
        
        Args:
            builder_name: Name of the builder to find
            
        Returns:
            Path to the builder file or None if not found
        """
        # Strategy 1: Try standard naming convention first
        standard_path = self.builders_dir / f"builder_{builder_name}_step.py"
        if standard_path.exists():
            return str(standard_path)
        
        # Strategy 2: Use FlexibleFileResolver for known patterns and fuzzy matching
        flexible_path = self.flexible_resolver.find_builder_file(builder_name)
        if flexible_path and Path(flexible_path).exists():
            return flexible_path
        
        # Strategy 3: Return None if nothing found
        return None
    
    def find_config_file(self, builder_name: str) -> Optional[str]:
        """
        Hybrid config file resolution with production registry integration.
        
        Priority:
        1. Production registry mapping: script_name -> canonical_name -> config_name
        2. Standard pattern: config_{builder_name}_step.py
        3. FlexibleFileResolver patterns (includes fuzzy matching)
        
        Args:
            builder_name: Name of the builder to find config for
            
        Returns:
            Path to the config file or None if not found
        """
        # Strategy 1: Use production registry mapping
        try:
            canonical_name = self._get_canonical_step_name(builder_name)
            config_base_name = self._get_config_name_from_canonical(canonical_name)
            registry_path = self.configs_dir / f"config_{config_base_name}_step.py"
            if registry_path.exists():
                return str(registry_path)
        except Exception:
            # Continue with other strategies if registry mapping fails
            pass
        
        # Strategy 2: Try standard naming convention
        standard_path = self.configs_dir / f"config_{builder_name}_step.py"
        if standard_path.exists():
            return str(standard_path)
        
        # Strategy 3: Use FlexibleFileResolver for known patterns and fuzzy matching
        flexible_path = self.flexible_resolver.find_config_file(builder_name)
        if flexible_path and Path(flexible_path).exists():
            return flexible_path
        
        # Strategy 4: Return None if nothing found
        return None
    
    def _get_canonical_step_name(self, script_name: str) -> str:
        """
        Convert script name to canonical step name using production registry logic.
        
        This uses the same approach as Level-3 validator to ensure consistency
        with the production system's mapping logic.
        
        Args:
            script_name: Script name (e.g., 'mims_package', 'model_evaluation_xgb')
            
        Returns:
            Canonical step name (e.g., 'Package', 'XGBoostModelEval')
        """
        # Import here to avoid circular imports
        try:
            from ....registry.step_names import get_step_name_from_spec_type
        except ImportError:
            # Fallback if registry is not available
            return self._fallback_canonical_name(script_name)
        
        # Convert script name to spec_type format (same as Level-3)
        parts = script_name.split('_')
        
        # Handle job type variants
        job_type_suffixes = ['training', 'validation', 'testing', 'calibration']
        job_type = None
        base_parts = parts
        
        if len(parts) > 1 and parts[-1] in job_type_suffixes:
            job_type = parts[-1]
            base_parts = parts[:-1]
        
        # Convert to PascalCase for spec_type
        spec_type_base = ''.join(word.capitalize() for word in base_parts)
        
        if job_type:
            spec_type = f"{spec_type_base}_{job_type.capitalize()}"
        else:
            spec_type = spec_type_base
        
        # Use production function to get canonical name (strips job type suffix)
        try:
            canonical_name = get_step_name_from_spec_type(spec_type)
            return canonical_name
        except Exception:
            # Fallback: return the base spec_type without job type suffix
            return spec_type_base
    
    def _fallback_canonical_name(self, script_name: str) -> str:
        """
        Fallback method to generate canonical name when registry is unavailable.
        
        Args:
            script_name: Script name
            
        Returns:
            Canonical step name
        """
        parts = script_name.split('_')
        
        # Handle job type variants
        job_type_suffixes = ['training', 'validation', 'testing', 'calibration']
        base_parts = parts
        
        if len(parts) > 1 and parts[-1] in job_type_suffixes:
            base_parts = parts[:-1]
        
        # Convert to PascalCase
        return ''.join(word.capitalize() for word in base_parts)
    
    def _get_config_name_from_canonical(self, canonical_name: str) -> str:
        """
        Get config file base name from canonical step name using production registry.
        
        Uses the STEP_NAMES registry to find the config class name,
        then derives the config file name from that.
        
        Args:
            canonical_name: Canonical step name (e.g., 'Package', 'XGBoostModelEval')
            
        Returns:
            Config file base name (e.g., 'package', 'model_eval_step_xgboost')
        """
        # Import here to avoid circular imports
        try:
            from ....registry.step_names import STEP_NAMES
        except ImportError:
            # Fallback if registry is not available
            return self._fallback_config_name(canonical_name)
        
        # Get config class name from STEP_NAMES registry
        if canonical_name in STEP_NAMES:
            config_class = STEP_NAMES[canonical_name]['config_class']
            
            # Convert config class name to file name
            # e.g., 'PackageConfig' -> 'package'
            # e.g., 'XGBoostModelEvalConfig' -> 'model_eval_step_xgboost'
            
            # Remove 'Config' suffix
            if config_class.endswith('Config'):
                base_name = config_class[:-6]  # Remove 'Config'
            else:
                base_name = config_class
            
            # Convert CamelCase to snake_case
            snake_case = self._camel_to_snake(base_name)
            return snake_case
        
        # Fallback if not in registry
        return self._fallback_config_name(canonical_name)
    
    def _fallback_config_name(self, canonical_name: str) -> str:
        """
        Fallback method to generate config name when registry is unavailable.
        
        Args:
            canonical_name: Canonical step name
            
        Returns:
            Config file base name
        """
        return self._camel_to_snake(canonical_name)
    
    def _camel_to_snake(self, name: str) -> str:
        """
        Convert CamelCase to snake_case.
        
        Args:
            name: CamelCase string
            
        Returns:
            snake_case string
        """
        # Insert underscores before uppercase letters
        snake_case = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        snake_case = re.sub('([a-z0-9])([A-Z])', r'\1_\2', snake_case).lower()
        return snake_case
    
    def get_resolution_diagnostics(self, builder_name: str) -> Dict[str, Any]:
        """
        Get detailed diagnostics about file resolution attempts.
        
        Args:
            builder_name: Name of the builder
            
        Returns:
            Dictionary containing resolution diagnostics
        """
        diagnostics = {
            'builder_name': builder_name,
            'builder_resolution': {},
            'config_resolution': {},
            'available_files': {}
        }
        
        # Builder file resolution diagnostics
        builder_standard = self.builders_dir / f"builder_{builder_name}_step.py"
        builder_flexible = self.flexible_resolver.find_builder_file(builder_name)
        
        diagnostics['builder_resolution'] = {
            'standard_path': str(builder_standard),
            'standard_exists': builder_standard.exists(),
            'flexible_path': builder_flexible,
            'flexible_exists': builder_flexible and Path(builder_flexible).exists(),
            'final_result': self.find_builder_file(builder_name)
        }
        
        # Config file resolution diagnostics
        config_standard = self.configs_dir / f"config_{builder_name}_step.py"
        config_flexible = self.flexible_resolver.find_config_file(builder_name)
        
        try:
            canonical_name = self._get_canonical_step_name(builder_name)
            config_base_name = self._get_config_name_from_canonical(canonical_name)
            config_registry = self.configs_dir / f"config_{config_base_name}_step.py"
        except Exception as e:
            canonical_name = "error"
            config_base_name = "error"
            config_registry = None
        
        diagnostics['config_resolution'] = {
            'canonical_name': canonical_name,
            'config_base_name': config_base_name,
            'registry_path': str(config_registry) if config_registry else None,
            'registry_exists': config_registry.exists() if config_registry else False,
            'standard_path': str(config_standard),
            'standard_exists': config_standard.exists(),
            'flexible_path': config_flexible,
            'flexible_exists': config_flexible and Path(config_flexible).exists(),
            'final_result': self.find_config_file(builder_name)
        }
        
        # Available files report
        diagnostics['available_files'] = self.flexible_resolver.get_available_files_report()
        
        return diagnostics

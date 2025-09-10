"""
Dynamic file resolution for alignment validation.

Provides intelligent file discovery and matching capabilities to find
component files (contracts, specs, builders, configs) for scripts.
"""

import re
import difflib
from typing import Dict, Optional, Any
from pathlib import Path


class FlexibleFileResolver:
    """
    Dynamic file resolution with file-system-driven discovery.
    
    This resolver discovers actual files in the filesystem and matches them
    to script names using intelligent pattern matching, eliminating the need
    for hardcoded mappings that become stale.
    """
    
    def __init__(self, base_directories: Dict[str, str]):
        """
        Initialize the file resolver with base directories.
        
        Args:
            base_directories: Dictionary mapping component types to their base directories
                             e.g., {'contracts': 'src/cursus/steps/contracts', ...}
        """
        self.base_dirs = {k: Path(v) for k, v in base_directories.items()}
        self.file_cache = {}  # Cache discovered files
        self._discover_all_files()
    
    def _discover_all_files(self):
        """Discover all actual files in each directory and extract base names."""
        for component_type, directory in self.base_dirs.items():
            self.file_cache[component_type] = self._scan_directory(directory, component_type)
    
    def _scan_directory(self, directory: Path, component_type: str) -> Dict[str, str]:
        """
        Scan directory and extract base names from actual files.
        
        Args:
            directory: Directory path to scan
            component_type: Type of component (contracts, specs, builders, configs)
            
        Returns:
            Dict mapping base_names to actual filenames
        """
        file_map = {}
        
        if not directory.exists():
            return file_map
        
        # Define patterns for each component type
        patterns = {
            'scripts': r'^(.+)\.py$',
            'contracts': r'^(.+)_contract\.py$',
            'specs': r'^(.+)_spec\.py$',
            'builders': r'^builder_(.+)_step\.py$',
            'configs': r'^config_(.+)_step\.py$'
        }
        
        pattern = patterns.get(component_type)
        if not pattern:
            return file_map
        
        regex = re.compile(pattern)
        
        for file_path in directory.glob('*.py'):
            if file_path.name.startswith('__'):
                continue
                
            match = regex.match(file_path.name)
            if match:
                base_name = match.group(1)
                file_map[base_name] = file_path.name
        
        return file_map
    
    def _find_best_match(self, script_name: str, component_type: str) -> Optional[str]:
        """
        Find best matching file for script name using multiple strategies.
        
        Args:
            script_name: Name of the script to find files for
            component_type: Type of component to search for
            
        Returns:
            Full path to the best matching file or None
        """
        available_files = self.file_cache.get(component_type, {})
        
        if not available_files:
            return None
        
        # Strategy 1: Exact match (case-sensitive)
        if script_name in available_files:
            return str(self.base_dirs[component_type] / available_files[script_name])
        
        # Strategy 2: Case-insensitive exact match
        script_lower = script_name.lower()
        for base_name, filename in available_files.items():
            if base_name.lower() == script_lower:
                return str(self.base_dirs[component_type] / filename)
        
        # Strategy 3: Normalized matching
        normalized_script = self._normalize_name(script_name)
        for base_name, filename in available_files.items():
            if self._normalize_name(base_name) == normalized_script:
                return str(self.base_dirs[component_type] / filename)
        
        # Strategy 4: Partial matching (contains)
        for base_name, filename in available_files.items():
            if script_lower in base_name.lower() or base_name.lower() in script_lower:
                return str(self.base_dirs[component_type] / filename)
        
        # Strategy 5: Fuzzy matching with lower threshold for better matching
        best_match = None
        best_score = 0.0
        
        for base_name, filename in available_files.items():
            # Try both directions for similarity
            score1 = self._calculate_similarity(script_name.lower(), base_name.lower())
            score2 = self._calculate_similarity(normalized_script, self._normalize_name(base_name))
            score = max(score1, score2)
            
            if score > 0.6 and score > best_score:  # Lower threshold for better matching
                best_score = score
                best_match = str(self.base_dirs[component_type] / filename)
        
        return best_match
    
    def _normalize_name(self, name: str) -> str:
        """
        Normalize names for better matching.
        
        Handles common variations:
        - preprocess vs preprocessing
        - eval vs evaluation  
        - xgb vs xgboost
        - dashes to underscores
        - dots to underscores
        
        Args:
            name: Name to normalize
            
        Returns:
            Normalized name
        """
        # Convert to lowercase
        normalized = name.lower()
        
        # Convert dashes to underscores
        normalized = normalized.replace('-', '_')
        # Replace dots with underscores, but preserve .py extension
        if normalized.endswith('.py'):
            # Remove .py, replace dots, then add .py back
            base_name = normalized[:-3]  # Remove .py
            base_name = base_name.replace('.', '_')
            normalized = base_name + '.py'
        else:
            # No .py extension, replace all dots
            normalized = normalized.replace('.', '_')
        
        # Handle common word variations
        variations = {
            'preprocess': 'preprocessing',
            'eval': 'evaluation',
            'xgb': 'xgboost',
        }
        
        for short, long in variations.items():
            # Handle both directions
            if short in normalized and long not in normalized:
                normalized = normalized.replace(short, long)
        
        return normalized
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate similarity between two strings using difflib.
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity ratio between 0.0 and 1.0
        """
        return difflib.SequenceMatcher(None, str1, str2).ratio()
    
    def refresh_cache(self):
        """Refresh file cache to pick up new files."""
        self._discover_all_files()
    
    def get_available_files_report(self) -> Dict[str, Dict[str, Any]]:
        """Get report of all discovered files for debugging."""
        report = {}
        for component_type, file_map in self.file_cache.items():
            report[component_type] = {
                'directory': str(self.base_dirs[component_type]),
                'files': list(file_map.values()),
                'base_names': list(file_map.keys()),
                'count': len(file_map)
            }
        return report
    
    def find_contract_file(self, script_name: str) -> Optional[str]:
        """
        Find contract file using dynamic discovery.
        
        Args:
            script_name: Name of the script (without .py extension)
            
        Returns:
            Path to the contract file or None if not found
        """
        return self._find_best_match(script_name, 'contracts')
    
    def find_spec_file(self, script_name: str) -> Optional[str]:
        """
        Find specification file using dynamic discovery.
        
        Args:
            script_name: Name of the script (without .py extension)
            
        Returns:
            Path to the specification file or None if not found
        """
        return self._find_best_match(script_name, 'specs')
    
    def find_specification_file(self, script_name: str) -> Optional[str]:
        """
        Alias for find_spec_file to maintain compatibility with existing code.
        
        Args:
            script_name: Name of the script (without .py extension)
            
        Returns:
            Path to the specification file or None if not found
        """
        return self.find_spec_file(script_name)
    
    def find_builder_file(self, script_name: str) -> Optional[str]:
        """
        Find builder file using dynamic discovery.
        
        Args:
            script_name: Name of the script (without .py extension)
            
        Returns:
            Path to the builder file or None if not found
        """
        return self._find_best_match(script_name, 'builders')
    
    def find_config_file(self, script_name: str) -> Optional[str]:
        """
        Find config file using dynamic discovery.
        
        Args:
            script_name: Name of the script (without .py extension)
            
        Returns:
            Path to the config file or None if not found
        """
        return self._find_best_match(script_name, 'configs')
    
    def find_all_component_files(self, script_name: str) -> Dict[str, Optional[str]]:
        """
        Find all component files for a given script.
        
        Args:
            script_name: Name of the script (without .py extension)
            
        Returns:
            Dictionary mapping component types to their file paths
        """
        return {
            'contract': self.find_contract_file(script_name),
            'spec': self.find_spec_file(script_name),
            'builder': self.find_builder_file(script_name),
            'config': self.find_config_file(script_name),
        }
    
    def extract_base_name_from_spec(self, spec_path: Path) -> str:
        """
        Extract the base name from a specification file path.
        
        For job type variant specifications like 'preprocessing_training_spec.py',
        this extracts 'preprocessing'.
        
        Args:
            spec_path: Path to the specification file
            
        Returns:
            Base name for the specification
        """
        stem = spec_path.stem  # Remove .py extension
        
        # Remove '_spec' suffix
        if stem.endswith('_spec'):
            stem = stem[:-5]
        
        # Remove job type suffix if present
        job_types = ['training', 'validation', 'testing', 'calibration']
        for job_type in job_types:
            if stem.endswith(f'_{job_type}'):
                return stem[:-len(job_type)-1]  # Remove _{job_type}
        
        return stem
    
    def find_spec_constant_name(self, script_name: str, job_type: str = 'training') -> Optional[str]:
        """
        Find the expected specification constant name for a script and job type.
        
        Args:
            script_name: Name of the script
            job_type: Job type variant (training, validation, testing, calibration)
            
        Returns:
            Expected constant name or None
        """
        # Generate based on discovered spec file patterns
        spec_file = self.find_spec_file(script_name)
        if spec_file:
            base_name = self.extract_base_name_from_spec(Path(spec_file))
            return f"{base_name.upper()}_{job_type.upper()}_SPEC"
        
        # Fallback to script name
        return f"{script_name.upper()}_{job_type.upper()}_SPEC"
